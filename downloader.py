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

    def get_papers_acl(self, conference: str, year: int) -> List[Tuple[str, str]]:
        """Get paper information from ACL Anthology"""
        base_url = f"https://aclanthology.org/events/{conference.lower()}-{year}/"
        response = self.make_request(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        papers = []
        for paper in soup.select('p.d-sm-flex'):
            pdf_link = paper.select_one('a[href$=".pdf"]')
            if pdf_link:
                title = paper.select_one('strong').text.strip()
                pdf_url = urljoin("https://aclanthology.org", pdf_link['href'])
                papers.append((title, pdf_url))
        return papers

    def download_conference_papers(
        self,
        papers: List[Tuple[str, str]],
        dir_path: str,
        conference: str,
        year: int,
        max_workers: int = 3  # Reduced number of concurrent downloads
    ):
        """Download papers with controlled concurrency"""
        failed_downloads = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {
                executor.submit(
                    self.download_file,
                    pdf_url,
                    os.path.join(dir_path, f"{self.sanitize_filename(title)}.pdf")
                ): (title, pdf_url) for title, pdf_url in papers
            }
            
            with tqdm(total=len(papers), desc=f"{conference} {year}") as pbar:
                for future in concurrent.futures.as_completed(future_to_paper):
                    title, url = future_to_paper[future]
                    try:
                        future.result()
                    except Exception as e:
                        failed_downloads.append((title, url))
                        logger.error(f"Failed to download {title}: {str(e)}")
                    pbar.update(1)
        
        # Report failed downloads
        if failed_downloads:
            logger.warning(f"\nFailed to download {len(failed_downloads)} papers from {conference} {year}:")
            for title, url in failed_downloads:
                logger.warning(f"- {title}: {url}")

    def download_acl(self, conference: str, year: int):
        """Download papers from ACL Anthology (ACL, EMNLP)"""
        dir_path = self.create_directory(conference, year)
        try:
            papers = self.get_papers_acl(conference, year)
            self.download_conference_papers(papers, dir_path, conference, year)
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