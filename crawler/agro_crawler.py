# agro_crawler.py

import requests
from bs4 import BeautifulSoup
import yaml
import hashlib
import time
from urllib.parse import urljoin, urlparse
from pathlib import Path
import logging

# CONFIG
START_URLS = [
    "https://www.fao.org/aquastat/fr/countries-and-basins/country/BFA/",
    "https://www.reca-niger.org/spip.php?article772",
    "https://www.inera.bf",  # Vérifier la structure réelle de ce site
    "https://fagri-burkina.com",
    "https://cimmyt.org"  # global, filtrer le contenu pertinent
]
ALLOWED_DOMAINS = ["fao.org", "reca-niger.org", "inera.bf", "fagri-burkina.com", "cimmyt.org"]
INDEX_FILE = "indexed_documents.yaml"
MAX_DEPTH = 2
USER_AGENT = "Mozilla/5.0 (compatible; AgroBot/1.0)"
PDF_EXTENSIONS = [".pdf"]

# LOGGER SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agro_crawler")

# Load existing index (if any)
def load_index():
    if Path(INDEX_FILE).exists():
        with open(INDEX_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_index(index):
    with open(INDEX_FILE, 'w') as f:
        yaml.dump(index, f)

# Generate unique hash of URL for indexing
def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()

# Determine if link is a PDF
def is_pdf_link(link):
    return any(link.lower().endswith(ext) for ext in PDF_EXTENSIONS)

# Crawl one page
def crawl_page(url, domain, visited, index, depth):
    if depth > MAX_DEPTH or url in visited:
        return
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        visited.add(url)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link['href']
        abs_url = urljoin(url, href)
        parsed = urlparse(abs_url)
        if parsed.netloc and parsed.netloc not in domain:
            continue  # out of domain

        if is_pdf_link(abs_url):
            uid = hash_url(abs_url)
            if uid not in index:
                logger.info(f"New PDF found: {abs_url}")
                index[uid] = {
                    'url': abs_url,
                    'title': link.get_text(strip=True) or "Untitled",
                    'source': parsed.netloc,
                    'found_at': url,
                    'timestamp': int(time.time())
                }
        else:
            crawl_page(abs_url, domain, visited, index, depth + 1)

# MAIN CRAWL
if __name__ == "__main__":
    index = load_index()
    visited = set()
    for start_url in START_URLS:
        logger.info(f"Scanning {start_url} ...")
        parsed_start = urlparse(start_url)
        domain = [parsed_start.netloc] + ALLOWED_DOMAINS
        crawl_page(start_url, domain, visited, index, 0)
    save_index(index)
    logger.info(f"Done. Indexed {len(index)} documents.")