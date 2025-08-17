import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urldefrag, urljoin

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


def _extract_meta(soup: BeautifulSoup) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    title = soup.find('title')
    if title and title.text:
        meta['title'] = title.text.strip()
    for attr in ['article:published_time', 'og:published_time', 'date', 'pubdate']:
        tag = soup.find('meta', attrs={'property': attr}) or soup.find('meta', attrs={'name': attr})
        if tag and tag.get('content'):
            meta['published_at'] = tag['content']
            break
    return meta


def clean_html_to_text(html_body: str, url: str) -> Tuple[str, Dict[str, Any]]:
    soup = BeautifulSoup(html_body, 'lxml')
    for sel in ['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript']:
        for el in soup.select(sel):
            el.decompose()

    for tag in soup.find_all():
        if tag.name in ['svg']:
            tag.decompose()

    meta = _extract_meta(soup)
    meta['source'] = url

    body = soup if soup.body is None else soup.body

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

    text = '\n'.join(lines)
    text = ' '.join(text.split())

    text = text.replace('\n ', '\n')
    return text.strip(), meta


def parse_sitemap(sitemap_url: str, max_links: int = 20) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    hash_manager = HashManager()

    try:
        r = requests.get(sitemap_url, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        logging.error(f'Не удалось загрузить sitemap: {e}')
        return results

    sm = BeautifulSoup(r.content, 'xml')
    url_items = []
    for u in sm.find_all('url'):
        loc = u.find('loc')
        pr = u.find('priority')
        if loc:
            priority = float(pr.text) if pr and pr.text else 0.5
            url_items.append({'url': loc.text.strip(), 'priority': priority})

    url_items.sort(key=lambda x: x['priority'], reverse=True)
    checked = 0
    i = 0
    while checked < max_links and i < len(url_items):
        page_url, _ = urldefrag(url_items[i]['url'])
        i += 1
        try:
            pr = requests.get(page_url, timeout=12)
            pr.raise_for_status()
            ps = BeautifulSoup(pr.content, 'lxml')

            for selector in ['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript']:
                for e in ps.select(selector):
                    e.decompose()

            body = ps.body if ps.body else ps
            text_for_hash = body.get_text(strip=True)
            if not text_for_hash or len(text_for_hash) < 100:
                continue

            if hash_manager.has_changed(page_url, text_for_hash):
                logging.info(f'Обновление/новая страница: {page_url}')
                results.append({'url': page_url, 'html_body': str(body)})
                checked += 1
        except requests.RequestException as e:
            logging.warning(f'Ошибка загрузки {page_url}: {e}')

    hash_manager.save_hashes()
    logging.info(f'Найдено новых/изменённых: {len(results)}')
    return results


if __name__ == '__main__':
    base = 'https://delprof.ru'
    sm = find_sitemap_url(base)
    if not sm:
        print('Sitemap не найден.')
    else:
        pages = parse_sitemap(sm, max_links=5)
        print('Страниц для индексации:', len(pages))
