# agro_full_crawler.py
"""
Crawler avancé pour AgroRAG avec gestion intelligente des ressources et contraintes temporelles.
"""

import requests
from bs4 import BeautifulSoup
import yaml
import hashlib
import time
import re
from urllib.parse import urljoin, urlparse, unquote
from pathlib import Path
import logging
from typing import List
from dataclasses import dataclass
from collections import deque
import signal
import sys


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger("agro_crawler")

@dataclass
class CrawlResult:
    """Structure pour stocker les résultats de crawling"""
    documents_found: int = 0  
    urls_indexed: int = 0
    new_urls_discovered: int = 0
    total_urls_processed: int = 0
    execution_time: float = 0.0
    stopped_gracefully: bool = True

class TimeBoundCrawler:
    """Crawler avec gestion du temps d'exécution"""    
    def __init__(self, max_execution_minutes: int = 5):
        self.max_execution_time = max_execution_minutes * 60  
        self.start_time = time.time()
        self.should_stop = False
        self.grace_period = 30  

        try:
            self.base_dir = Path(__file__).parent.parent
        except NameError:
            self.base_dir = Path.cwd()
            if self.base_dir.name != "agroRAG" and self.base_dir.name == "crawler":
                self.base_dir = self.base_dir.parent
        
        self.crawler_dir = self.base_dir / "crawler"
        self.data_dir = self.base_dir / "data"   

        self.url_config_file = self.crawler_dir / "URLCrawlerConf.yaml"
        self.corpus_config_file = self.crawler_dir / "CORPUSCrawlerConf.yaml"
        self.pdf_index_file = self.data_dir / "indexed_documents.yaml"
        self.url_index_file = self.data_dir / "indexed_urls.yaml"

        # Structures de données
        self.url_config = {}
        self.corpus_config = {}
        self.pdf_index = {}
        self.url_index = {}
        self.discovered_urls = set()
        self.visited_urls = set()
        self.url_queue = deque()
        self.already_explored_urls = set()
        self.stats = CrawlResult()
        
        # Configuration du signal d'arrêt gracieux
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self):
        """Gestionnaire de signal pour arrêt gracieux"""
        self.should_stop = True
    
    def _check_time_limit(self) -> bool:
        """Vérifie si on approche de la limite de temps"""
        elapsed = time.time() - self.start_time
        remaining = self.max_execution_time - elapsed      
        if remaining <= self.grace_period:
            if not self.should_stop:
                self.should_stop = True
            return True
        return False
    
    def load_configurations(self) -> None:
        """Charge les configurations YAML"""
        try:
            # Configuration des URLs
            if self.url_config_file.exists():
                with open(self.url_config_file, 'r', encoding='utf-8') as f:
                    self.url_config = yaml.safe_load(f) or {}
            else:
                self.url_config = {"base_urls": [], "discovered_urls": [], "manual_urls": []}
            
            # Configuration du corpus
            if self.corpus_config_file.exists():
                with open(self.corpus_config_file, 'r', encoding='utf-8') as f:
                    self.corpus_config = yaml.safe_load(f) or {}
            else:
                self.corpus_config = {"primary_keywords": [], "secondary_keywords": []}
            
            # Index des PDFs
            if self.pdf_index_file.exists():
                with open(self.pdf_index_file, 'r', encoding='utf-8') as f:
                    self.pdf_index = yaml.safe_load(f) or {}
            else:
                self.pdf_index = {}
            
            # Index des URLs
            if self.url_index_file.exists():
                with open(self.url_index_file, 'r', encoding='utf-8') as f:
                    self.url_index = yaml.safe_load(f) or {}
            else:
                self.url_index = {}
            self._load_already_explored_urls()
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des configurations: {e}")
            raise
    
    def _clean_discovered_urls(self) -> None:
        """Supprime les URLs déjà explorées"""
        current_discovered = self.url_config.get("discovered_urls", [])
        cleaned_discovered = [url for url in current_discovered if url not in self.already_explored_urls]
        self.url_config["discovered_urls"] = cleaned_discovered
        

    def save_configurations(self) -> None:
        """Sauvegarde des configurations et index"""
        try:
            self._clean_discovered_urls()
            if self.discovered_urls:
                existing_discovered = set(self.url_config.get("discovered_urls", []))
                new_discovered = self.discovered_urls - existing_discovered - self.already_explored_urls
                if new_discovered:
                    self.url_config.setdefault("discovered_urls", []).extend(list(new_discovered))
            
            with open(self.url_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.url_config, f, default_flow_style=False, allow_unicode=True)
            with open(self.pdf_index_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.pdf_index, f, default_flow_style=False, allow_unicode=True)
            with open(self.url_index_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.url_index, f, default_flow_style=False, allow_unicode=True)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def _hash_url(self, url: str) -> str:
        """Génère un hash unique pour une URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _load_already_explored_urls(self) -> None:
        """Charge les URLs déjà explorées"""
        # Charger depuis l'index des URLs
        for url_data in self.url_index.values():
            self.already_explored_urls.add(url_data['url'])
        
        # Charger depuis l'index des PDFs
        for pdf_data in self.pdf_index.values():
            self.already_explored_urls.add(pdf_data['found_at'])
        
        # Charger depuis URLs explorées non indexées
        explored_file = self.crawler_dir / "explored_urls.txt"
        if explored_file.exists():
            try:
                with open(explored_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        url = line.strip()
                        if url:
                            self.already_explored_urls.add(url)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement des URLs explorées: {e}")


    def _save_explored_url(self, url: str) -> None:
        """Sauvegarde une URL explorée (non indexée)"""
        # Éviter les doublons
        if url in self.already_explored_urls:
            return
        explored_file = self.crawler_dir / "explored_urls.txt"
        try:
            with open(explored_file, 'a', encoding='utf-8') as f:
                f.write(f"{url}\n")
            self.already_explored_urls.add(url)
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde de l'URL explorée: {e}")


    def _get_urls_to_process(self) -> List[str]:
        """Retourne la liste des URLs à traiter en filtrant celles déjà explorées"""
        all_urls = []
        
        # Collecter toutes les URLs sources
        base_urls = self.url_config.get("base_urls", [])
        discovered_urls = self.url_config.get("discovered_urls", [])
        manual_urls = self.url_config.get("manual_urls", [])
        
        # Filtrer les URLs déjà explorées
        new_base_urls = [url for url in base_urls if url not in self.already_explored_urls]
        new_discovered_urls = [url for url in discovered_urls if url not in self.already_explored_urls]
        new_manual_urls = [url for url in manual_urls if url not in self.already_explored_urls]
        
        # Assembler la liste finale
        all_urls.extend(new_base_urls)
        all_urls.extend(new_discovered_urls) 
        all_urls.extend(new_manual_urls)
        
        return all_urls
    
    def _is_valid_url(self, url: str) -> bool:
        """Vérifie si une URL est valide pour le crawling"""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return False
            
            # Vérifier les domaines bloqués 
            blocked_domains = self.url_config.get("blocked_domains", [])
            if any(domain in parsed.netloc for domain in blocked_domains):
                return False
            return True
            
        except Exception:
            return False
    
    def _evaluate_content_quality(self, text_content: str, html_content: BeautifulSoup) -> dict:
        """
        Évalue la qualité (métriques) du contenu d'une page web.
        """
        # Métriques de base
        metrics = {
            'text_length': len(text_content),
            'word_count': len(text_content.split()),
            'has_meaningful_content': False,
            'text_to_html_ratio': 0.0,
            'substantive_content_score': 0
        }
        
        # Calculer le ratio texte/HTML 
        html_length = len(str(html_content))
        if html_length > 0:
            metrics['text_to_html_ratio'] = len(text_content) / html_length
        
        # Extraction de texte des paragraphes 
        paragraphs = html_content.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        substantive_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        metrics['substantive_word_count'] = len(substantive_text.split())
        
        # Évaluer si le contenu est significatif
        min_word_count = self.corpus_config.get("quality_thresholds", {}).get("minimum_word_count", 1000)
        min_substantive_count = self.corpus_config.get("quality_thresholds", {}).get("minimum_substantive_words", 500)
        
        if (metrics['word_count'] >= min_word_count and 
            metrics['substantive_word_count'] >= min_substantive_count):
            metrics['has_meaningful_content'] = True
        metrics['substantive_content_score'] = min(10, metrics['substantive_word_count'] // 20)
        
        return metrics
        
    def _is_document_link(self, url: str) -> bool:
        """Vérifie si le lien pointe vers un document téléchargeable"""
        url_lower = url.lower()
        doc_extensions_list = self.corpus_config.get("document_filtering", {}).get("document_extensions", [])
        if not doc_extensions_list:
            doc_extensions = ('.pdf', '.doc', '.docx')
        else: 
            doc_extensions = tuple(doc_extensions_list)
            
        return url_lower.endswith(doc_extensions)
        
    def _get_document_type(self, url: str) -> str:
        """Renvoie le type de document basé sur l'extension"""
        url_lower = url.lower()
        extension_mapping = {
            '.pdf': "PDF",
            '.doc': "DOC",
            '.docx': "DOCX",
            '.ppt': "PPT", 
            '.pptx': "PPT",
            '.xls': "XLS", 
            '.xlsx': "XLS",
            '.odt': "ODT",
            '.ods': "ODS",
            '.odp': "ODP"
        }
        
        # Vérifier chaque extension dans le mapping
        for ext, doc_type in extension_mapping.items():
            if url_lower.endswith(ext):
                return doc_type
                
        return "Document"
    
    def _calculate_content_score(self, text: str, title: str = "") -> int:
        """Calcule un score de pertinence basé sur les mots-clés"""
        text_lower = text.lower()
        title_lower = title.lower()
        score = 0
        
        # Mots-clés principaux
        primary_keywords = self.corpus_config.get("primary_keywords", [])
        primary_weight = self.corpus_config.get("scoring", {}).get("primary_keyword_weight", 10)
        
        for keyword in primary_keywords:
            keyword_lower = keyword.lower()
            # Double score si trouvé dans le titre
            if keyword_lower in title_lower:
                score += primary_weight * 2
            elif keyword_lower in text_lower:
                score += primary_weight
        
        # Mots-clés secondaires
        secondary_keywords = self.corpus_config.get("secondary_keywords", [])
        secondary_weight = self.corpus_config.get("scoring", {}).get("secondary_keyword_weight", 5)
        
        for keyword in secondary_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower:
                score += secondary_weight * 2
            elif keyword_lower in text_lower:
                score += secondary_weight
        
        # Expressions exactes
        exact_phrases = self.corpus_config.get("exact_phrases", [])
        exact_weight = self.corpus_config.get("scoring", {}).get("exact_phrase_weight", 15)
        
        for phrase in exact_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in title_lower:
                score += exact_weight * 2
            elif phrase_lower in text_lower:
                score += exact_weight
        
        return score
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extrait et filtre les liens d'une page"""
        links = []
        
        for link in soup.find_all("a", href=True):
            href = link.get('href', '').strip()
            if not href or href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                continue
            abs_url = urljoin(base_url, href)
            if self._is_valid_url(abs_url) and abs_url not in self.visited_urls:
                links.append(abs_url)
        
        return links
    
    def _crawl_page(self, url: str, depth: int = 0) -> None:
        """Crawl une page spécifique"""
        if self.should_stop or self._check_time_limit():
            return
        if url in self.visited_urls:
            return
        
        try:
            # Configuration de la requête
            config = self.url_config.get("crawl_config", {})
            timeout = config.get("request_timeout", 15)
            user_agent = config.get("user_agent", "Mozilla/5.0 (compatible; AgroBot/2.0)")
            delay = config.get("delay_between_requests", 1)
            
            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "fr,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
            
            # Faire la requête
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            self.visited_urls.add(url)
            self.stats.total_urls_processed += 1
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraire le titre et le texte
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ""
            
            # Extraire le texte principal
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
            if main_content:
                text_content = main_content.get_text(strip=True)
            else:
                text_content = soup.get_text(strip=True)
            
            # Évaluer la qualité du contenu
            content_metrics = self._evaluate_content_quality(text_content, soup)
            score = self._calculate_content_score(text_content, title_text)
            if content_metrics['substantive_content_score'] > 0:
                score += content_metrics['substantive_content_score']
            
            # Seuils d'acceptation
            threshold = self.corpus_config.get("scoring", {}).get("minimum_score_threshold", 8)
            min_words = self.corpus_config.get("quality_thresholds", {}).get("minimum_word_count", 1000)
            min_text_ratio = self.corpus_config.get("quality_thresholds", {}).get("minimum_text_ratio", 0.5)
            url_has_value = False
            
            # Indexer l'URL si pertinente ET significatif
            if (score >= threshold and 
                content_metrics['word_count'] >= min_words and
                content_metrics['text_to_html_ratio'] >= min_text_ratio and
                content_metrics['has_meaningful_content']):
                
                url_hash = self._hash_url(url)
                if url_hash not in self.url_index:
                    self.url_index[url_hash] = {
                        'url': url,
                        'title': title_text,
                        'score': score,
                        'depth': depth,
                        'timestamp': int(time.time()),
                        'content_length': content_metrics['substantive_word_count'],
                        'domain': urlparse(url).netloc,
                        'quality_metrics': {
                            'word_count': content_metrics['word_count'],
                            'text_ratio': round(content_metrics['text_to_html_ratio'], 3)
                        }
                    }
                    self.stats.urls_indexed += 1
                    url_has_value = True  # Cette URL a du contenu de qualité
                    logger.info(f"URL indexée (score: {score}, mots: {content_metrics['substantive_word_count']}): {title_text[:100]}...")

              
            # Extraire et traiter les liens
            links = self._extract_links(soup, url)
            doc_links = [link for link in links if self._is_document_link(link)]
            for doc_url in doc_links:
                doc_hash = self._hash_url(doc_url)
                if doc_hash not in self.pdf_index:                   
                    is_relevant = False
                    score_document = 0
                    
                    # Charger les paramètres de filtrage depuis la configuration
                    doc_filter_config = self.corpus_config.get("document_filtering", {})
                    min_score_required = doc_filter_config.get("minimum_score_required", 15)
                    keywords_found = []
                    
                    # Les poids pour différents critères
                    url_weight = doc_filter_config.get("url_keyword_weight", 10)
                    primary_kw_weight = doc_filter_config.get("primary_keyword_weight", 15)
                    secondary_kw_weight = doc_filter_config.get("secondary_keyword_weight", 5)
                    page_quality_weight = doc_filter_config.get("page_quality_weight", 12)
                    exact_phrase_weight = doc_filter_config.get("exact_phrase_weight", 20)

                    # Filtrage par URL du document
                    for kw in self.corpus_config.get("primary_keywords", []):
                        if kw.lower() in doc_url.lower():
                            score_document += url_weight
                            keywords_found.append(f"URL:{kw}")
                    
                    # Filtrage par expressions exactes dans l'URL
                    for phrase in self.corpus_config.get("exact_phrases", []):
                        if phrase.lower() in doc_url.lower():
                            score_document += exact_phrase_weight
                            keywords_found.append(f"URL-Exact:{phrase}")

                    # Filtrage par texte du lien 
                    for link_elem in soup.find_all('a', href=True):
                        if link_elem.get('href', '').strip() and doc_url.endswith(link_elem.get('href', '').split('/')[-1]):
                            link_text = link_elem.get_text(strip=True).lower()
                            
                            # Vérifier les mots-clés principaux
                            for kw in self.corpus_config.get("primary_keywords", []):
                                if kw.lower() in link_text:
                                    score_document += primary_kw_weight
                                    keywords_found.append(f"Lien-P:{kw}")
                            
                            # Vérifier les mots-clés secondaires
                            for kw in self.corpus_config.get("secondary_keywords", []):
                                if kw.lower() in link_text:
                                    score_document += secondary_kw_weight
                                    keywords_found.append(f"Lien-S:{kw}")
                            
                            # Vérifier les expressions exactes (bonus plus important)
                            for phrase in self.corpus_config.get("exact_phrases", []):
                                if phrase.lower() in link_text:
                                    score_document += exact_phrase_weight
                                    keywords_found.append(f"Lien-Exact:{phrase}")
                            break  

                    # Filtrage par pertinence de la page source
                    if url_has_value:
                        if score >= threshold * 1.5: 
                            score_document += page_quality_weight
                            keywords_found.append("Page_Qualité")
                        elif score >= threshold:
                            score_document += page_quality_weight // 2
                            keywords_found.append("Page_Pertinente")
                    
                    # Décision finale basée sur le score total
                    if score_document >= min_score_required:
                        is_relevant = True
                        logger.info(f"Document pertinent (score: {score_document})")
                    else:
                        logger.debug(f"Document ignoré (score insuffisant: {score_document}/{min_score_required})")

                    if is_relevant:
                        doc_type = self._get_document_type(doc_url)
                        filename = doc_url.split('/')[-1]
                        try:
                            filename = unquote(filename)
                        except:
                            pass
                        doc_title = filename
                        for link_elem in soup.find_all('a', href=True):
                            if link_elem.get('href', '').strip() and doc_url.endswith(link_elem.get('href', '').split('/')[-1]):
                                link_text = link_elem.get_text(strip=True)
                                if link_text and not link_text.startswith('Document '):
                                    doc_title = link_text
                                    break
                        
                        self.pdf_index[doc_hash] = {
                            'url': doc_url,
                            'title': doc_title,
                            'type': doc_type,
                            'source': urlparse(url).netloc,
                            'found_at': url,
                            'timestamp': int(time.time())
                        }
                        self.stats.documents_found += 1
                        url_has_value = True  
              
            # Sauvegarder dans explored_urls.txt 
            if url_has_value:
                self._save_explored_url(url)
            else:
                self.visited_urls.add(url)
              
            # Ajouter les nouvelles URLs découvertes pour exploration future
            max_depth = self.url_config.get("crawl_config", {}).get("max_depth", 3)
            if depth <= max_depth:
                page_links = [link for link in links if not self._is_document_link(link)]
                for link in page_links:  
                    if link not in self.visited_urls:
                        self.discovered_urls.add(link)
                        self.url_queue.append((link, depth + 1))
                        self.stats.new_urls_discovered += 1
            
            # Respecter le délai entre requêtes
            time.sleep(delay)
            
        except Exception as e:
            logger.warning(f"Erreur lors du crawl de {url}")

    def run(self) -> CrawlResult:
        """Exécute le crawling avec contraintes temporelles"""
        
        self.start_time = time.time()
        try:
            self.load_configurations()
            urls_to_process = self._get_urls_to_process()
            for url in urls_to_process:
                if self._is_valid_url(url):
                    self.url_queue.append((url, 0))
            
            urls_processed = 0
            while self.url_queue and not self.should_stop:
                if self._check_time_limit():
                    break
                url, depth = self.url_queue.popleft()
                if url not in self.visited_urls:
                    self._crawl_page(url, depth)                    
                    urls_processed += 1
                    if urls_processed % 10 == 0:
                        elapsed = time.time() - self.start_time
                        logger.info(f"Progression: {urls_processed} URLs traitées en {elapsed:.2f}s")
            
            # Sauvegarder les résultats
            self.save_configurations()
            
            # Calculer les statistiques finales
            self.stats.execution_time = time.time() - self.start_time
            self.stats.stopped_gracefully = not self._check_time_limit()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Erreur fatale pendant le crawling: {e}")
            self.stats.stopped_gracefully = False
            raise
        
        finally:
            try:
                self.save_configurations()
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde finale: {e}")


def main(max_time=5):

    crawler = TimeBoundCrawler(max_execution_minutes=max_time)

    try:
        result = crawler.run()
    
        # Résumé des résultats
        print(f"\nRésultat du crawl :")
        print(f"  - URLs indexées : {result.urls_indexed}")
        print(f"  - Documents trouvés : {result.documents_found}")
        print(f"  - Nouvelles URLs découvertes : {result.new_urls_discovered}")
        print(f"  - Total URLs traitées : {result.total_urls_processed}")
        print(f"  - Temps d'exécution : {result.execution_time:.2f}s")
        print(f"  - Arrêt gracieux : {result.stopped_gracefully}")
        
    except KeyboardInterrupt:
        return None

    except Exception as e:
        raise

if __name__ == "__main__":
    main(max_time=1)
