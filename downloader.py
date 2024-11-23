import requests
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import concurrent.futures
import re
from tqdm import tqdm
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from ratelimit import limits, sleep_and_retry
from typing import Tuple, List
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
import hashlib
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conference_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_openai_client(base_url: str = "http://localhost:8080/v1") -> openai.Client:
    """Setup OpenAI client with local endpoint"""
    return openai.Client(
        base_url=base_url,
        api_key="not-needed"  # Local server might not need API key
    )

@dataclass
class Paper:
    title: str
    url: str
    abstract: str
    conference: str
    year: int
    is_relevant: bool = False
    downloaded: bool = False
    file_path: str = ""

class PaperFilter:
    def __init__(self, cache_db="paper_cache.db", base_url="http://localhost:8080/v1"):
        self.client = setup_openai_client(base_url)
        self.cache_db = cache_db
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_cache (
                    paper_hash TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    is_relevant BOOLEAN,
                    cached_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS downloaded_papers (
                    paper_hash TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    file_path TEXT,
                    conference TEXT,
                    year INTEGER,
                    downloaded_at TIMESTAMP
                )
            """)

    def _get_paper_hash(self, title: str, abstract: str) -> str:
        """Generate unique hash for paper based on title and abstract"""
        return hashlib.sha256(f"{title}{abstract}".encode()).hexdigest()

    def get_cached_relevance(self, title: str, abstract: str) -> tuple[bool, bool]:
        """Get cached relevance decision. Returns (is_cached, is_relevant)"""
        paper_hash = self._get_paper_hash(title, abstract)
        with sqlite3.connect(self.cache_db) as conn:
            result = conn.execute(
                "SELECT is_relevant FROM paper_cache WHERE paper_hash = ?",
                (paper_hash,)
            ).fetchone()
            return bool(result is not None), bool(result[0]) if result else False

    def cache_relevance(self, title: str, abstract: str, is_relevant: bool):
        """Cache the relevance decision"""
        paper_hash = self._get_paper_hash(title, abstract)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO paper_cache 
                (paper_hash, title, abstract, is_relevant, cached_at)
                VALUES (?, ?, ?, ?, ?)
            """, (paper_hash, title, abstract, is_relevant, datetime.now()))

    def is_already_downloaded(self, title: str, url: str) -> tuple[bool, str]:
        """Check if paper is already downloaded. Returns (is_downloaded, file_path)"""
        paper_hash = self._get_paper_hash(title, url)
        with sqlite3.connect(self.cache_db) as conn:
            result = conn.execute(
                "SELECT file_path FROM downloaded_papers WHERE paper_hash = ?",
                (paper_hash,)
            ).fetchone()
            return bool(result is not None), result[0] if result else ""

    def mark_as_downloaded(self, paper: Paper):
        """Mark paper as downloaded in the database"""
        paper_hash = self._get_paper_hash(paper.title, paper.url)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO downloaded_papers 
                (paper_hash, title, url, file_path, conference, year, downloaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (paper_hash, paper.title, paper.url, paper.file_path,
                 paper.conference, paper.year, datetime.now()))

    def is_relevant_paper(self, title: str, abstract: str) -> bool:
        """Check if paper is relevant using cache first, then LLM if needed"""
        is_cached, is_relevant = self.get_cached_relevance(title, abstract)
        
        if is_cached:
            logger.debug(f"Using cached relevance for: {title}")
            return is_relevant

        prompt = f"""Based on the following paper title and abstract, determine if it's primarily related to any of these topics:
Select Topics:
- Information Retrieval systems and techniques
- Language models (architecture, training, evaluation)
- Language Generation
- RLHF (Reinforcement Learning from Human Feedback)

Ignore Topics:
- Task-oriented Dialogue Systems
- Machine Translation

Title: {title}
Abstract: {abstract}

Respond with only "yes" if the paper is primarily about any of these topics, or "no" if it isn't. 
Response format: Just 'yes' or 'no'"""

        try:
            response = self.client.chat.completions.create(
                model="meta-llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )
            is_relevant = response.choices[0].message.content.strip().lower() == "yes"
            self.cache_relevance(title, abstract, is_relevant)
            return is_relevant
        except Exception as e:
            logger.error(f"Error classifying paper {title}: {str(e)}")
            return False

class ConferenceDownloader:
    def __init__(self, base_dir="papers"):
        self.base_dir = base_dir
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # Rate limits: 1 request per 10 seconds per domain
        self.domain_timestamps = {}
        self.min_delay = 10.0  # increased to 10 seconds between requests

    def get_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting"""
        from urllib.parse import urlparse
        return urlparse(url).netloc

    @sleep_and_retry
    @limits(calls=1, period=10)  # 1 call every 10 seconds
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, IOError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def make_request(self, url: str, stream: bool = False) -> requests.Response:
        """Make a rate-limited request with retries"""
        domain = self.get_domain(url)
        
        # Ensure minimum delay between requests to same domain
        current_time = time.time()
        if domain in self.domain_timestamps:
            elapsed = current_time - self.domain_timestamps[domain]
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
        
        response = self.session.get(url, headers=self.headers, stream=stream)
        response.raise_for_status()
        
        self.domain_timestamps[domain] = time.time()
        return response

    def create_directory(self, conference: str, year: int) -> str:
        """Create directory if it doesn't exist"""
        dir_path = os.path.join(self.base_dir, conference, str(year))
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(IOError)
    )
    def download_file(self, url: str, filepath: str) -> bool:
        """Download a single file with retry logic"""
        try:
            response = self.make_request(url, stream=True)
            
            # Get total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=os.path.basename(filepath)
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up partial download
            raise

    def get_papers_acl(self, conference: str, year: int) -> List[Paper]:
        """Get paper information from ACL Anthology with abstracts"""
        base_url = f"https://aclanthology.org/events/{conference.lower()}-{year}/"
        response = self.make_request(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        papers = []
        paper_filter = PaperFilter()
        
        for paper in soup.select('p.d-sm-flex'):
            try:
                pdf_link = paper.select_one('a[href$=".pdf"]')
                if pdf_link:
                    title = paper.select_one('strong').text.strip()
                    pdf_url = urljoin("https://aclanthology.org", pdf_link['href'])
                    abstract = paper.find_next("div").text.strip()
                    
                    # Check if already downloaded
                    is_downloaded, file_path = paper_filter.is_already_downloaded(title, pdf_url)
                    if is_downloaded:
                        logger.info(f"Skipping already downloaded paper: {title}")
                        continue
                    
                    # Check relevance
                    if paper_filter.is_relevant_paper(title, abstract):
                        papers.append(Paper(
                            title=title,
                            url=pdf_url,
                            abstract=abstract,
                            conference=conference,
                            year=year
                        ))
                        logger.info(f"Selected paper: {title}")
                    else:
                        logger.debug(f"Skipped non-relevant paper: {title}")
                    
                    time.sleep(self.min_delay)
                    
            except Exception as e:
                logger.error(f"Error processing paper: {str(e)}")
                continue
        
        return papers

    def download_conference_papers(
        self,
        papers: List[Paper],
        dir_path: str,
        conference: str,
        year: int,
        max_workers: int = 3
    ):
        """Download papers with controlled concurrency and save metadata"""
        paper_filter = PaperFilter()
        failed_downloads = []
        papers_to_download = []
        
        # Filter out already downloaded papers and prepare metadata
        metadata_entries = []
        for paper in papers:
            is_downloaded, file_path = paper_filter.is_already_downloaded(paper.title, paper.url)
            if is_downloaded:
                paper.downloaded = True
                paper.file_path = file_path
                logger.info(f"Paper already downloaded: {paper.title}")
            else:
                paper.file_path = os.path.join(dir_path, f"{self.sanitize_filename(paper.title)}.pdf")
                papers_to_download.append(paper)
            
            metadata_entries.append({
                "title": paper.title,
                "url": paper.url,
                "abstract": paper.abstract,
                "conference": conference,
                "year": year,
                "file_path": paper.file_path,
                "downloaded": paper.downloaded
            })
        
        # Save/update metadata
        metadata_file = os.path.join(dir_path, "metadata.jsonl")
        with open(metadata_file, "w", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Download new papers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {
                executor.submit(
                    self.download_file,
                    paper.url,
                    paper.file_path
                ): paper for paper in papers_to_download
            }
            
            with tqdm(total=len(papers_to_download), desc=f"{conference} {year}") as pbar:
                for future in concurrent.futures.as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    try:
                        success = future.result()
                        if success:
                            paper.downloaded = True
                            paper_filter.mark_as_downloaded(paper)
                    except Exception as e:
                        failed_downloads.append(paper)
                        logger.error(f"Failed to download {paper.title}: {str(e)}")
                    pbar.update(1)
    
    # def download_conference_papers(
    #     self,
    #     papers: List[Tuple[str, str]],
    #     dir_path: str,
    #     conference: str,
    #     year: int,
    #     max_workers: int = 3  # Reduced number of concurrent downloads
    # ):
    #     """Download papers with controlled concurrency"""
    #     failed_downloads = []
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         path_to_paper = lambda title: os.path.join(dir_path, f"{self.sanitize_filename(title)}.pdf")
    #         future_to_paper = {
    #             executor.submit(
    #                 self.download_file,
    #                 pdf_url,
    #                 path_to_paper(title)
    #             ): (title, pdf_url) for title, pdf_url, abstract in papers if not os.path.exists(path_to_paper(title))
    #         }
            
    #         with tqdm(total=len(papers), desc=f"{conference} {year}") as pbar:
    #             for future in concurrent.futures.as_completed(future_to_paper):
    #                 title, url = future_to_paper[future]
    #                 try:
    #                     future.result()
    #                 except Exception as e:
    #                     failed_downloads.append((title, url))
    #                     logger.error(f"Failed to download {title}: {str(e)}")
    #                 pbar.update(1)
        
    #     # Report failed downloads
    #     if failed_downloads:
    #         logger.warning(f"\nFailed to download {len(failed_downloads)} papers from {conference} {year}:")
    #         for title, url in failed_downloads:
    #             logger.warning(f"- {title}: {url}")

    def download_acl(self, conference: str, year: int):
        """Download papers from ACL Anthology (ACL, EMNLP)"""
        dir_path = self.create_directory(conference, year)
        try:
            papers = self.get_papers_acl(conference, year)
            self.download_conference_papers(papers, dir_path, conference, year, max_workers=1)
        except Exception as e:
            logger.error(f"Error processing {conference} {year}: {str(e)}")

    def download_neurips(self, year: int):
        """Download papers from NeurIPS"""
        # Similar implementation for NeurIPS with rate limiting...
        pass

    def download_icml(self, year: int):
        """Download papers from ICML"""
        # Similar implementation for ICML with rate limiting...
        pass

    def download_jmlr(self, year: int):
        """Download papers from JMLR"""
        # Similar implementation for JMLR with rate limiting...
        pass

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Convert title to valid filename"""
        return re.sub(r'[<>:"/\\|?*]', '', filename)[:200]  # Limit filename length

def main():
    # Create download directory
    base_dir = "ml_papers"
    os.makedirs(base_dir, exist_ok=True)
    
    # Initialize downloader
    downloader = ConferenceDownloader(base_dir=base_dir)
    
    # Configure which conferences and years to download
    conferences = {
        'ACL': list(range(2023, 2025)),
        'EMNLP': list(range(2023, 2025)),
        # 'NeurIPS': list(range(2020, 2024)),
        # 'ICML': list(range(2020, 2024)),
        # 'JMLR': list(range(2020, 2024))
    }
    
    try:
        for conf, years in conferences.items():
            for year in years:
                logger.info(f"Starting download for {conf} {year}")
                if conf in ['ACL', 'EMNLP']:
                    downloader.download_acl(conf, year)
                elif conf == 'NeurIPS':
                    downloader.download_neurips(year)
                elif conf == 'ICML':
                    downloader.download_icml(year)
                elif conf == 'JMLR':
                    downloader.download_jmlr(year)
                logger.info(f"Completed download for {conf} {year}")
                
                # Add delay between conference downloads
                time.sleep(5)
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        logger.info("Download process completed")

if __name__ == "__main__":
    main()