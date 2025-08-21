import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urldefrag, urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup

from config import HASH_STORE_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_sitemap_url(base_url: str) -> Optional[str]:
    logging.info(f'Поиск карты сайта для {base_url}...')
    robots_url = urljoin(base_url, '/robots.txt')
    try:
        r = requests.get(robots_url, timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith('sitemap:'):
                    sm = line.split(':', 1)[1].strip()
                    logging.info(f'Карта сайта найдена в robots.txt: {sm}')
                    return sm
    except requests.RequestException:
        logging.warning('Не удалось получить robots.txt')

    sitemap_url = urljoin(base_url, '/sitemap.xml')
    try:
        h = requests.head(sitemap_url, timeout=10)
        if h.status_code == 200:
            logging.info(f'Карта сайта найдена по умолчанию: {sitemap_url}')
            return sitemap_url
    except requests.RequestException:
        pass

    logging.error(f'Карта сайта не найдена для {base_url}')
    return None


class HashManager:
    def __init__(self, storage_path: Path = HASH_STORE_PATH):
        self.storage_path = Path(storage_path)
        self.hashes = self._load_hashes()
        logging.info(f'Загружено хэшей: {len(self.hashes)}')

    def _load_hashes(self) -> Dict[str, str]:
        if not self.storage_path.exists():
            return {}
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            logging.warning('Файл хэшей повреждён. Начинаем заново.')
            return {}

    def save_hashes(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.hashes, f, indent=2, ensure_ascii=False)

    def has_changed(self, url: str, content: str) -> bool:
        cur = hashlib.md5(content.encode('utf-8')).hexdigest()
        prev = self.hashes.get(url)
        if prev == cur:
            return False
        self.hashes[url] = cur
        return True


async def fetch(session: aiohttp.ClientSession, url: str, timeout: int = 12) -> Optional[str]:
    try:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.text()
    except Exception as e:
        logging.warning(f'Ошибка загрузки {url}: {e}')
        return None


def clean_html_to_text(html_body: str, url: str) -> Tuple[str, Dict[str, Any]]:
    soup = BeautifulSoup(html_body, 'lxml')
    for sel in ['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript']:
        for el in soup.select(sel):
            el.decompose()

    meta = {'source': url}
    title = soup.find('title')
    if title and title.text:
        meta['title'] = title.text.strip()

    body = soup.body if soup.body else soup
    lines: List[str] = []
    for el in body.descendants:
        if getattr(el, 'name', None) in ['h1', 'h2', 'h3']:
            text = el.get_text(' ', strip=True)
            if text:
                lines.append('\n' + text + '\n')
        elif getattr(el, 'name', None) in ['p', 'li']:
            text = el.get_text(' ', strip=True)
            if text:
                lines.append(text)

    text = ' '.join(' '.join(lines).split())
    text = text.replace('\n ', '\n')
    return text.strip(), meta


async def parse_sitemap(sitemap_url: str, max_links: int = 20) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    hash_manager = HashManager()

    async with aiohttp.ClientSession() as session:
        sitemap_text = await fetch(session, sitemap_url, timeout=15)
        if not sitemap_text:
            logging.error(f'Не удалось загрузить sitemap: {sitemap_url}')
            return results

        sm = BeautifulSoup(sitemap_text, 'xml')
        url_items = []
        for u in sm.find_all('url'):
            loc = u.find('loc')
            pr = u.find('priority')
            if loc:
                priority = float(pr.text) if pr and pr.text else 0.5
                url_items.append({'url': loc.text.strip(), 'priority': priority})

        url_items.sort(key=lambda x: x['priority'], reverse=True)

        for item in url_items:
            if len(results) >= max_links:
                logging.info(f"Достигнут лимит в {max_links} новых/измененных страниц.")
                break

            page_url = item['url']
            url_clean, _ = urldefrag(page_url)
            html = await fetch(session, url_clean)
            if not html:
                continue

            text_for_hash, _ = clean_html_to_text(html, url_clean)
            if len(text_for_hash) < 100:
                continue

            if hash_manager.has_changed(url_clean, text_for_hash):
                logging.info(f'Обновление/новая страница: {url_clean}')
                results.append({'url': url_clean, 'html_body': html})

    hash_manager.save_hashes()
    logging.info(f'Найдено новых/изменённых страниц: {len(results)}')
    return results


if __name__ == '__main__':
    async def main():
        base = 'https://delprof.ru'
        sm = find_sitemap_url(base)
        if not sm:
            print('Sitemap не найден.')
        else:
            pages = await parse_sitemap(sm, max_links=10)
            print('Страниц для индексации:', len(pages))


    asyncio.run(main())
