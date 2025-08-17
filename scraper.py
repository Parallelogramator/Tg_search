import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_sitemap_url(base_url: str) -> Optional[str]:
    """
    Автоматически находит URL карты сайта.

    Сначала проверяет файл /robots.txt, затем стандартный адрес /sitemap.xml.

    Args:
        base_url: Корневой URL сайта (например, "https://delprof.ru").

    Returns:
        Найденный URL карты сайта или None, если найти не удалось.
    """
    logging.info(f"Поиск карты сайта для {base_url}...")

    robots_url = urljoin(base_url, "/robots.txt")
    try:
        response = requests.get(robots_url, timeout=10)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    logging.info(f"Карта сайта найдена в robots.txt: {sitemap_url}")
                    return sitemap_url
    except requests.RequestException as e:
        logging.warning(f"Не удалось проверить robots.txt: {e}")

    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        response = requests.head(sitemap_url, timeout=10)
        if response.status_code == 200:
            logging.info(f"Карта сайта найдена по стандартному адресу: {sitemap_url}")
            return sitemap_url
    except requests.RequestException as e:
        logging.warning(f"Не удалось проверить стандартный адрес sitemap.xml: {e}")

    logging.error(f"Не удалось автоматически найти карту сайта для {base_url}")
    return None


class HashManager:
    """Класс для управления хранилищем хэшей страниц."""

    def __init__(self, storage_path: str = "hash_storage.json"):
        self.storage_path = Path(storage_path)
        self.hashes = self._load_hashes()
        logging.info(f"Загружено {len(self.hashes)} существующих хэшей из {self.storage_path}")

    def _load_hashes(self) -> Dict[str, str]:
        """Загружает хэши из JSON-файла. Возвращает пустой словарь, если файл не найден."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            logging.warning(f"Файл {self.storage_path} поврежден. Будет создан новый.")
            return {}

    def save_hashes(self):
        """Сохраняет текущие хэши в JSON-файл."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.hashes, f, indent=4, ensure_ascii=False)
        logging.info(f"Хэши ({len(self.hashes)} записей) успешно сохранены в {self.storage_path}")

    def has_changed(self, url: str, content: str) -> bool:
        """
        Проверяет, изменился ли контент страницы.
        Обновляет хэш, если страница новая или изменилась.

        Returns:
            True, если страница новая или изменилась (требуется обработка).
            False, если страница не изменилась (можно пропустить).
        """
        current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        stored_hash = self.hashes.get(url)

        if stored_hash == current_hash:
            return False
        else:
            self.hashes[url] = current_hash
            return True


def parse_sitemap(sitemap_url: str, max_links: int = 20) -> list[dict[str, str | Any]] | None:
    """
    Парсит карту сайта, обрабатывая только новые или измененные страницы.

    Args:
        sitemap_url: Полный URL к файлу sitemap.xml.
        max_links: Максимальное количество самых приоритетных ссылок для проверки.

    Returns:
        Список словарей с URL и HTML-кодом только для новых или обновленных страниц.
    """
    results = []
    hash_manager = HashManager()

    try:
        logging.info(f"Загрузка карты сайта: {sitemap_url}")
        response = requests.get(sitemap_url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Не удалось загрузить карту сайта. Ошибка: {e}")
        return results

    sitemap_soup = BeautifulSoup(response.content, "xml")

    url_data = []
    for url_entry in sitemap_soup.find_all("url"):
        loc_tag = url_entry.find("loc")
        priority_tag = url_entry.find("priority")
        if loc_tag:
            priority = float(priority_tag.text) if priority_tag and priority_tag.text else 0.5
            url_data.append({"url": loc_tag.text.strip(), "priority": priority})

    urls_to_check = sorted(url_data, key=lambda x: x['priority'], reverse=True)
    logging.info(f"Найдено {len(urls_to_check)} ссылок. Проверка топ-{max_links} по приоритету.")

    links = 0
    i = 0
    while links < max_links and i < len(urls_to_check):
        page_url, _ = urldefrag(urls_to_check[i]['url'])
        try:
            page_response = requests.get(page_url, timeout=10)
            page_response.raise_for_status()

            page_soup = BeautifulSoup(page_response.content, "lxml")

            for selector in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
                for element in page_soup.select(selector):
                    element.decompose()

            body = page_soup.body
            if not body:
                logging.warning(f"Тег <body> не найден на странице {page_url}. Пропускаем.")
                continue

            page_text = body.get_text(separator=" ", strip=True)

            if hash_manager.has_changed(page_url, page_text):
                logging.info(f"  -> Обнаружены изменения на странице: {page_url}. Добавляем в обработку.")
                results.append({
                    "url": page_url,
                    "html_body": str(body)
                })
                links += 1
            else:
                logging.info(f"  -> Страница не изменилась: {page_url}. Пропускаем.")

        except requests.RequestException as e:
            logging.error(f"   Не удалось обработать страницу {page_url}. Ошибка: {e}")
        finally:
            i += 1

    hash_manager.save_hashes()

    logging.info(f"Обработка завершена. Найдено {len(results)} новых/обновленных страниц.")
    return results


if __name__ == "__main__":
    TARGET_SITEMAP_URL = "https://delprof.ru"

    print("--- Первый запуск (или после удаления hash_storage.json) ---")
    pages_data_first_run = parse_sitemap(TARGET_SITEMAP_URL, max_links=5)
    print(f"\nНа первом запуске обработано: {len(pages_data_first_run)} страниц.\n")

    print("\n--- Второй запуск (имитация обновления без изменений на сайте) ---")
    pages_data_second_run = parse_sitemap(TARGET_SITEMAP_URL, max_links=5)
    print(f"\nНа втором запуске обработано: {len(pages_data_second_run)} страниц.\n")

    print("Проверьте файл 'hash_storage.json', который появился в папке проекта.")
