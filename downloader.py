import requests
import random
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
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import argparse

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('conference_downloader.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(level="INFO")])
logger = logging.getLogger('rich')

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

"""
python downloader.py
-bu --base_url: Base URL for LLM API
-m --model: LLM model name
-db --cache_db: SQLite database for caching

Example:
LM Studio >> python downloader.py --base_url http://localhost:8080/v1 --model meta-llama-3.1-8b-instruct --cache_db paper_cache.db
vllm >> python downloader.py --base_url http://localhost:8000/v1 --model meta-llama/Meta-Llama-3.1-8B-Instruct --cache_db paper_cache.db
"""
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-bu", "--base_url", type=str, default="http://localhost:8080/v1", help="Base URL for LLM API")
arg_parser.add_argument("-m", "--model", type=str, default="meta-llama-3.1-8b-instruct", help="LLM model name")
arg_parser.add_argument("-db", "--cache_db", type=str, default="paper_cache.db", help="SQLite database for caching")
args = arg_parser.parse_args()


def setup_openai_client(base_url: str) -> openai.Client:
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
    def __init__(self, cache_db, base_url, model):
        self.client = setup_openai_client(base_url)
        self.cache_db = cache_db
        self._init_db()
        self.model = model

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
            # cache web url requests
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_cache (
                    url TEXT PRIMARY KEY,
                    content TEXT,
                    cached_at TIMESTAMP
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

    # def clear_relevance_entry(self, title: str, abstract: str):
    #     """Clear all relevance entry from cache"""
    #     paper_hash = self._get_paper_hash(title, abstract)
    #     with sqlite3.connect(self.cache_db) as conn:
    #         conn.execute(
    #             "DELETE FROM paper_cache WHERE paper_hash = ?",
    #             (paper_hash,)
    #         )

    def is_relevant_paper(self, title: str, abstract: str, prompt: str) -> bool:
        """Check if paper is relevant using cache first, then LLM if needed"""
        is_cached, is_relevant = self.get_cached_relevance(title, abstract)
        
        if is_cached:
            logger.debug(f"Using cached relevance for: {title}")
            return is_relevant

        prompt = prompt.format(title=title, abstract=abstract)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            is_relevant = response.choices[0].message.content.strip().lower() == "yes"
            self.cache_relevance(title, abstract, is_relevant)
            return is_relevant
        except Exception as e:
            logger.error(f"Error classifying paper {title}: {str(e)}")
            return False
        
    def get_cached_url_response(self, url: str) -> str:
        """Get cached response content for URL"""
        with sqlite3.connect(self.cache_db) as conn:
            result = conn.execute(
                "SELECT content FROM web_cache WHERE url = ?",
                (url,)
            ).fetchone()
            is_cached = bool(result is not None)
            return is_cached, result[0] if result else ""

    def cache_url_response(self, url: str, content: str):
        """Cache the response content for URL"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO web_cache 
                (url, content, cached_at)
                VALUES (?, ?, ?)
            """, (url, content, str(datetime.now())))


class ConferenceDownloader:
    def __init__(self, base_dir="papers", progress=None):
        self.base_dir = base_dir
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # Rate limits: 1 request per 10 seconds per domain
        self.domain_timestamps = {}
        self.min_delay = 10.0  # increased to 10 seconds between requests

        assert progress is not None
        self.progress = progress

        # A common filter for all needs
        self.db_filter = PaperFilter(args.cache_db, args.base_url, args.model)

    def get_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting"""
        from urllib.parse import urlparse
        return urlparse(url).netloc

    @sleep_and_retry
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, IOError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def make_request(self, url: str, stream: bool = False, cache_response: bool = True) -> requests.Response:
        """Make a rate-limited request with retries"""

        # Check if response is cached
        if cache_response:
            is_cached, content = self.db_filter.get_cached_url_response(url)
            if is_cached:
                j_response = json.loads(content)
                response = requests.Response()
                response.status_code = j_response['status_code']
                response.headers = j_response['headers']
                response._content = j_response['content'].encode()
                return response

        # sleep for a random time b/w (10, 20) seconds
        time.sleep(random.randint(10, 20))
        
        response = self.session.get(url, headers=self.headers, stream=stream)
        response.raise_for_status()
        
        # Cache response content
        if cache_response:
            serialized_response = json.dumps({
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text
            })
            self.db_filter.cache_url_response(url, serialized_response)
        
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
            # wait a random time b/w (10, 20)
            time.sleep(random.randint(10, 20))
            response = self.make_request(url, stream=True, cache_response=False)
            
            # Get total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
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
        paper_filter = PaperFilter(args.cache_db, args.base_url, args.model)

        # Get total number of papers for progress tracking
        all_papers = soup.select('p.d-sm-flex')
        task_filter = self.progress.add_task(f"[red]Filtering {conference} {year}", total=len(all_papers))

        # Selection prompt
        interestingness_prompt = open("prompts/acl_prompt.txt", "r").read()
        
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
                    
                    # Check relevance
                    if paper_filter.is_relevant_paper(title, abstract, interestingness_prompt):
                        papers.append(Paper(
                            title=title,
                            url=pdf_url,
                            abstract=abstract,
                            conference=conference,
                            year=year
                        ))
                        logger.info(f"Selected paper: {title}")
                    else:
                        logger.info(f"Skipped non-relevant paper: {title}")
            except Exception as e:
                logger.error(f"Error processing paper: {str(e)}")

            self.progress.update(task_filter, advance=1)
        
        return papers

    def get_papers_neurips(self, year: int) -> List[Paper]:
        """Get paper information from NeurIPS with abstracts"""
        base_url = f"https://neurips.cc/Conferences/{year}/Schedule"
        response = self.make_request(base_url)

        # Parse HTML and extract paper information
        soup = BeautifulSoup(response.text, 'html.parser')
        papers = []
        paper_filter = PaperFilter(args.cache_db, args.base_url, args.model)

        # tracker 
        all_papers = soup.select('div.maincard')
        task_filter = self.progress.add_task(f"[red]Filtering NeurIPS {year}", total=len(all_papers))

        # Selection prompt
        interestingness_prompt = open("prompts/neurips_prompt.txt", "r").read()

        for paper in soup.select('div.maincard'):
            try:
                # title in maincardbody, openreview link in a with title="OpenReview"
                title = paper.select_one('div.maincardBody').text.strip()

                # relevance cleaner
                # paper_filter.clear_relevance_entry(title, "<abstract not provided>")
                # paper_filter.clear_relevance_entry(title, "")
                # continue

                openreview_link = paper.select_one('a[title="OpenReview"]')
                if openreview_link:
                    forum_url = openreview_link['href']
                    pdf_url = forum_url.replace("forum", "pdf")

                    # filter-1: does the title seem interesting, novel and worth reading?
                    title_filter_pass = False
                    if paper_filter.is_relevant_paper(title, "abstract not provided", interestingness_prompt):
                        # logging.info(f"Passing title filter: {title}")
                        title_filter_pass = True
                    else:
                        logging.info(f"Failed title filter: {title}")

                    if title_filter_pass:
                        # TODO: need to avoid this repeated request, if abstract was already fetched
                        # get abstract from forum page
                        response_forum = self.make_request(forum_url)
                        soup_forum = BeautifulSoup(response_forum.text, 'html.parser')
                        # comes in pair of span.note-content-field and span.note-content-value
                        features = {f.text.strip():v.text.strip() for f, v in zip(soup_forum.select('strong.note-content-field'), soup_forum.select('span.note-content-value'))}
                        abstract = features.get("Abstract:", "")

                        # filter-2: from the abstract, does the paper seem incremental or do they talk about incremental results on some specific task? 
                        # we really want to read the papers that are possibly high impact and would be rewarding.
                        if paper_filter.is_relevant_paper(title, abstract, interestingness_prompt):
                            # paper_filter.clear_relevance_entry(title, abstract)
                            # continue
                            papers.append(Paper(
                                title=title,
                                url=pdf_url,
                                abstract=abstract,
                                conference="NeurIPS",
                                year=year
                            ))
                            logger.info(f"Selected paper: {title}")
                        else:
                            logger.info(f"Skipped non-relevant paper: {title}")
            except Exception as e:
                logger.error(f"Error processing paper: {str(e)}")

            self.progress.update(task_filter, advance=1)

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
        paper_filter = PaperFilter(args.cache_db, args.base_url, args.model)
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
        task_downloader = self.progress.add_task(f"[green]Downloading {conference} {year}", total=len(papers_to_download))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {
                executor.submit(
                    self.download_file,
                    paper.url,
                    paper.file_path
                ): paper for paper in papers_to_download
            }
            
            for future in concurrent.futures.as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    success = future.result()
                    if success:
                        paper.downloaded = True
                        paper_filter.mark_as_downloaded(paper)
                        logger.info(f"Downloaded {paper.title}")
                    else:
                        failed_downloads.append(paper)
                        logger.error(f"Failed to download {paper.title}")
                except Exception as e:
                    failed_downloads.append(paper)
                    logger.error(f"Failed to download {paper.title}: {str(e)}")

                self.progress.update(task_downloader, advance=1)
        
    
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
        dir_path = self.create_directory("NeurIPS", year)
        try:
            papers = self.get_papers_neurips(year)
            self.download_conference_papers(papers, dir_path, "NeurIPS", year)
        except Exception as e:
            logger.error(f"Error processing NeurIPS {year}: {str(e)}")

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
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
        downloader = ConferenceDownloader(base_dir=base_dir, progress=progress)
        
        # Configure which conferences and years to download
        conferences = {
            # 'EMNLP': list(range(2023, 2025)),
            'NeurIPS': list(range(2023, 2025)),
            # 'ACL': list(range(2023, 2025)),
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
