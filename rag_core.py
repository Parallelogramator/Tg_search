import os
import logging
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from scraper import parse_sitemap, find_sitemap_url


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class RAGInitializationError(Exception):
    """Собственная ошибка для критических проблем при инициализации RAG-системы."""
    pass


class RAGCore:
    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store_path = vector_store_path
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise RAGInitializationError(
                "Требуется GOOGLE_API_KEY в окружении. Установите переменную окружения и повторите запуск."
            )

        logging.info("Инициализация RAGCore в режиме Google GenAI.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        self.embeddings = self._setup_embeddings()
        self.llm = self._setup_llm()

        self.vector_store = self._load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        prompt_template = """
        Ты — вежливый и информативный ИИ-ассистент. Твоя задача — дать развернутый ответ на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте.
        Не придумывай информацию. Если в контексте нет ответа, так и скажи: "К сожалению, в моей базе знаний нет информации по этому вопросу".
        Ответ должен быть структурированным и содержать ссылки на источники в формате [источник].
        
        КОНТЕКСТ:
        {context}
        
        ВОПРОС:
        {question}
        
        ОТВЕТ:
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def _setup_embeddings(self):
        """Инициализирует модель эмбеддингов Google Generative AI."""
        logging.info("Инициализация эмбеддингов Google Generative AI.")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_QUERY")

    def _setup_llm(self):
        """Инициализирует ChatGoogleGenerativeAI (GenAI / Gemini) для генерации ответов."""
        logging.info("Инициализация ChatGoogleGenerativeAI (GenAI).")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            convert_system_message_to_human=True
        )

    def _load_or_create_vector_store(self):
        """Загружает существующее векторное хранилище или создаёт новое по указанному сайту."""
        if os.path.exists(self.vector_store_path):
            logging.info(f"Загрузка существующей векторной базы из {self.vector_store_path}")
            return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            logging.warning("Векторная база не найдена. Запускаем создание новой базы знаний (по умолчанию: delprof.ru).")
            return self.create_knowledge_base('https://delprof.ru', max_links=200)

    def _prepare_documents_from_url(self, site_url: str, max_links: int):
        """
        Вспомогательный метод для сбора, подготовки и разделения документов на чанки.
        Ожидается, что parse_sitemap возвращает список {'url':..., 'html_body':...}
        """
        sitemap_url = find_sitemap_url(site_url)
        if not sitemap_url:
            raise ValueError(f"Не удалось найти карту сайта для {site_url}.")

        pages_data = parse_sitemap(sitemap_url, max_links=max_links)
        if not pages_data:
            return [], 0

        documents = [
            Document(page_content=data['html_body'], metadata={"source": data['url']})
            for data in pages_data
        ]
        chunks = self.text_splitter.split_documents(documents)
        return chunks, len(pages_data)

    def create_knowledge_base(self, site_url: str, max_links: int = 200) -> FAISS:
        """
        Создает новую векторную базу знаний с нуля.
        """
        chunks, page_count = self._prepare_documents_from_url(site_url, max_links)

        if not chunks:
            raise RAGInitializationError(
                f"Не удалось собрать документы с сайта {site_url} для создания базы знаний."
            )

        logging.info(f"Создание векторной базы из {page_count} страниц ({len(chunks)} чанков)...")
        new_vector_store = FAISS.from_documents(chunks, self.embeddings)
        new_vector_store.save_local(self.vector_store_path)
        logging.info(f"Новая база знаний успешно создана и сохранена в {self.vector_store_path}")

        return new_vector_store

    def update_knowledge_base(self, site_url: str, max_links: int = 50) -> str:
        """
        Обновляет существующую базу знаний (upsert новых/изменённых чанков).
        """
        logging.info(f"Запуск обновления базы знаний с сайта: {site_url}")
        chunks, page_count = self._prepare_documents_from_url(site_url, max_links)

        if not chunks:
            logging.info("Новых или измененных страниц не найдено. База знаний актуальна.")
            return "База знаний уже актуальна. Новых страниц не найдено."

        logging.info(f"Добавление {len(chunks)} новых чанков в векторную базу...")
        self.vector_store.add_documents(chunks)
        self.vector_store.save_local(self.vector_store_path)
        logging.info("Векторная база успешно обновлена и сохранена.")

        return f"База знаний успешно обновлена. Добавлено {page_count} новых/измененных страниц."

    def get_answer(self, query: str):
        """Основная функция для получения ответа на запрос."""
        logging.info(f"Получен запрос: {query}")

        retrieved_docs = self.retriever.get_relevant_documents(query)

        if not retrieved_docs:
            logging.warning("Релевантные документы не найдены.")
            return "К сожалению, в моей базе знаний нет информации по этому вопросу.", []

        context_text = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source')}\n\n{doc.page_content}" for doc in retrieved_docs])
        sources = list({doc.metadata.get("source") for doc in retrieved_docs})

        logging.info("Генерация ответа с помощью Google GenAI...")
        answer_text = self.llm_chain.predict(context=context_text, question=query)

        return answer_text, sources


if __name__ == "__main__":
    try:
        rag = RAGCore(vector_store_path="vector_store")
    except RAGInitializationError as e:
        logging.error("Ошибка инициализации RAGCore: %s", e)
        raise SystemExit(1)

    test_query = "Какие есть меры поддержки для ИТ-компаний?"
    answer, answer_sources = rag.get_answer(test_query)

    print("\n--- РЕЗУЛЬТАТ ТЕСТА ---")
    print(f"Вопрос: {test_query}")
    print(f"\nОтвет:\n{answer}")
    print(f"\nИсточники: {answer_sources}")
    print("----------------------")
