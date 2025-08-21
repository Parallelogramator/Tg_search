import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import aiohttp
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import (
    USE_GOOGLE, GOOGLE_API_KEY, LOCAL_EMBEDDING_MODEL, LOCAL_LLM_MODEL,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FUSED, CHUNK_SIZE, CHUNK_OVERLAP,
    VECTOR_DIR, BM25_CORPUS_PATH, TEMPERATURE, MAX_TOKENS, DEFAULT_UPDATE_MAX_LINKS, DEFAULT_SITE
)
from scraper import parse_sitemap, find_sitemap_url, clean_html_to_text

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGInitializationError(Exception):
    pass



class HybridEmbeddings(Embeddings):
    """Переключаемые эмбеддинги: Google или локально."""
    def __init__(self):
        if USE_GOOGLE and GOOGLE_API_KEY:
            self.provider = "google"
            self._model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        else:
            self.provider = "local"
            self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "google":
            return self._model.embed_documents(texts)
        return [self._model.encode(t, convert_to_numpy=True).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.provider == "google":
            return self._model.embed_query(text)
        return self._model.encode(text, convert_to_numpy=True).tolist()



class HybridLLM:
    def __init__(self):
        if USE_GOOGLE and GOOGLE_API_KEY:
            self.mode = "google"
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=TEMPERATURE,
                convert_system_message_to_human=True
            )
            self.chain = None
        else:
            self.mode = "local"
            self.generator = pipeline(
                "text-generation",
                model=LOCAL_LLM_MODEL,
                device_map="auto",
                torch_dtype="auto"
            )

    def generate(self, prompt: str) -> str:
        if self.mode == "google":
            raise RuntimeError("Use LLMChain with Google LLM in RAGCore.")
        else:
            out = self.generator(prompt, max_new_tokens=MAX_TOKENS, do_sample=False, temperature=TEMPERATURE)
            return out[0]["generated_text"]


class RAGCore:
    def __init__(self, vector_store_path: str = str(VECTOR_DIR)):
        if USE_GOOGLE and not GOOGLE_API_KEY:
            raise RAGInitializationError("Указан режим Google, но GOOGLE_API_KEY отсутствует.")
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.retriever = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; "]
        )

        self.embeddings = HybridEmbeddings()
        self.llm_provider = HybridLLM()


        self._bm25 = None
        self._bm25_corpus_docs = []
        self._load_or_build_bm25_corpus()

        template = (
            "Ты — вежливый и точный ИИ-ассистент. Отвечай ТОЛЬКО по контексту, без фантазий.\n"
            "Если ответа нет в контексте — прямо скажи об этом.\n\n"
            "Формат ответа: краткое резюме, затем структурированные пункты, затем вывод.\n"
            "Добавь цитаты фраз (если уместно) и пометь источники в конце (я добавлю ссылки сам).\n\n"
            "КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}\n\nОТВЕТ:"
        )
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        if self.llm_provider.mode == "google":
            self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm_provider.llm)
        else:
            self.llm_chain = None

        self._last_updated = datetime.utcnow().isoformat(timespec='seconds')

    @classmethod
    async def create(cls, vector_store_path: str = str(VECTOR_DIR)):
        """Асинхронно создает и инициализирует экземпляр RAGCore."""
        instance = cls(vector_store_path)

        if os.path.exists(instance.vector_store_path):
            logger.info("Загрузка FAISS индекса...")
            instance.vector_store = FAISS.load_local(
                instance.vector_store_path, instance.embeddings, allow_dangerous_deserialization=True
            )
        else:
            logger.warning("FAISS индекс не найден. Инициализация на базе сайта по умолчанию.")
            await instance.async_load()

        instance.retriever = instance.vector_store.as_retriever(search_kwargs={"k": TOP_K_DENSE})
        instance._load_or_build_bm25_corpus()

        return instance

    async def async_load(self):
        self.vector_store = await self.create_knowledge_base(DEFAULT_SITE, DEFAULT_UPDATE_MAX_LINKS)

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
                text, meta = clean_html_to_text(html, url)
                if not text or len(text) < 200:
                    return None
                doc = Document(page_content=text, metadata=meta)
                return doc
        except Exception as e:
            logger.warning(f"Ошибка при загрузке {url}: {e}")
            return None

    async def _prepare_documents_from_url(self, site_url: str, max_links: int | None = 50) -> Tuple[
        List[Document], int]:
        sitemap_url = find_sitemap_url(site_url)
        if not sitemap_url:
            raise ValueError(f"Не удалось найти карту сайта для {site_url}")

        pages_data = await parse_sitemap(sitemap_url, max_links=max_links)
        if not pages_data:
            return [], 0

        docs: List[Document] = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_page(session, data["url"]) for data in pages_data]
            results = await asyncio.gather(*tasks)

        for doc in results:
            if doc:
                chunks = self.text_splitter.split_documents([doc])
                docs.extend(chunks)
        return docs, len(pages_data)

    async def create_knowledge_base(self, site_url: str, max_links: int | None = None) -> FAISS:
        docs, page_count = await self._prepare_documents_from_url(site_url, max_links)
        if not docs:
            raise RAGInitializationError(f"Не удалось собрать документы для {site_url}")

        logger.info(f"Создание FAISS индекса из {page_count} страниц, чанков: {len(docs)}")
        vs = FAISS.from_documents(docs, self.embeddings)
        vs.save_local(self.vector_store_path)
        self._save_bm25_corpus(docs)
        self._build_bm25(docs)
        self.vector_store = vs
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K_DENSE})
        self._last_updated = datetime.utcnow().isoformat(timespec='seconds')
        return vs

    async def update_knowledge_base(self, site_url: str, max_links: int | None = None) -> str:
        docs, page_count = await self._prepare_documents_from_url(site_url, max_links)
        if not docs:
            return "База знаний уже актуальна. Новых/изменённых страниц не найдено."

        self.vector_store.add_documents(docs)
        self.vector_store.save_local(self.vector_store_path)

        self._append_bm25_corpus(docs)
        self._build_bm25(self._bm25_corpus_docs)

        self._last_updated = datetime.utcnow().isoformat(timespec='seconds')
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K_DENSE})
        self._last_updated = datetime.utcnow().isoformat(timespec='seconds')
        return f"Готово: добавлено {page_count} страниц, чанков: {len(docs)}."


    def _load_or_build_bm25_corpus(self):
        if os.path.exists(BM25_CORPUS_PATH):
            try:
                with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f]
                docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in rows]
                self._build_bm25(docs)
                logger.info("BM25 корпус загружен.")
                return
            except Exception:
                logger.exception("Не удалось загрузить BM25 корпус. Перестроим.")

        all_docs = self.vector_store.docstore._dict.values()
        docs = list(all_docs)
        self._save_bm25_corpus(docs)
        self._build_bm25(docs)

    def _save_bm25_corpus(self, docs: List[Document]):
        with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

    def _append_bm25_corpus(self, docs: List[Document]):
        with open(BM25_CORPUS_PATH, "a", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

    def _build_bm25(self, docs: List[Document]):
        self._bm25_corpus_docs = docs
        tokenized = [d.page_content.split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 индекс построен. Документов: %d", len(docs))

    def _dense_retrieve(self, query: str, k: int) -> List[Document]:
        return self.retriever.get_relevant_documents(query)[:k]

    def _bm25_retrieve(self, query: str, k: int) -> List[Document]:
        if not self._bm25_corpus_docs:
            return []
        scores = self._bm25.get_scores(query.split())
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._bm25_corpus_docs[i] for i in idx]

    @staticmethod
    def _rrf_fusion(lists: List[List[Document]], k: int) -> List[Document]:
        K = 60
        score = {}
        for docs in lists:
            for rank, d in enumerate(docs):
                key = (d.page_content[:50], d.metadata.get("source", ""))
                score[key] = score.get(key, 0) + 1.0 / (K + rank + 1)
        doc_map = {}
        for docs in lists:
            for d in docs:
                key = (d.page_content[:50], d.metadata.get("source", ""))
                if key not in doc_map:
                    doc_map[key] = d
        ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, _ in ranked[:k]]

    def _maybe_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        from sentence_transformers import CrossEncoder
        try:
            reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            pairs = [[query, d.page_content] for d in docs]
            scores = reranker.predict(pairs).tolist()
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [docs[i] for i in order]
        except Exception:
            logger.warning("Rerank не выполнен (модель не доступна). Продолжаем без него.")
            return docs

    def _split_for_telegram(self, text: str, limit: int = 4096) -> list[str]:
        """Делит текст на части для Telegram (не разрывая слова)."""
        parts = []
        while len(text) > limit:
            split_pos = text.rfind("\n", 0, limit)
            if split_pos == -1:
                split_pos = text.rfind(" ", 0, limit)
            if split_pos == -1:
                split_pos = limit
            parts.append(text[:split_pos].strip())
            text = text[split_pos:].strip()
        if text:
            parts.append(text)
        return parts

    def get_answer(self, query: str) -> tuple[str, list[Any]] | tuple[list[str], list[dict[str, str]]]:
        dense_docs = self._dense_retrieve(query, TOP_K_DENSE)
        bm25_docs = self._bm25_retrieve(query, TOP_K_BM25)

        fused = self._rrf_fusion([dense_docs, bm25_docs], TOP_K_FUSED)

        if os.getenv("USE_RERANKER", "true").lower() == "true":
            fused = self._maybe_rerank(query, fused)

        seen_urls = set()
        context_blocks = []
        sources: List[Dict[str, str]] = []
        for d in fused:
            url = d.metadata.get("source") or d.metadata.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                title = d.metadata.get("title") or url
                sources.append({"title": title, "url": url})
            head = f"Источник: {d.metadata.get('title', '')}\nURL: {url}\n"
            context_blocks.append(head + d.page_content)

        if not context_blocks:
            return "К сожалению, в моей базе знаний нет информации по этому вопросу.", []

        context_text = "\n\n---\n\n".join(context_blocks)

        if self.llm_provider.mode == "google":
            answer = self.llm_chain.predict(context=context_text, question=query)
        else:
            prompt = self.prompt.format(context=context_text, question=query)
            answer = self.llm_provider.generate(prompt)

        messages = self._split_for_telegram(answer, 3900)

        return messages, sources

    def get_stats(self) -> Dict[str, str | int]:
        try:
            n_chunks = len(self.vector_store.docstore._dict)
        except Exception:
            n_chunks = 0
        try:
            sources = set()
            for d in self.vector_store.docstore._dict.values():
                sources.add(d.metadata.get("source", ""))
            n_docs = len(sources)
        except Exception:
            n_docs = 0
        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "last_updated": self._last_updated
        }
