#!/usr/bin/env python3
"""
Unified Data Manager for London Historical LLM
Consolidates all data collection, processing, and cleaning functionality
"""

import os
import sys
import json
import urllib.request
import time
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import unicodedata
import re

# BeautifulSoup for HTML processing
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# PDF processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Language detection
try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from sanitize_filenames import FilenameSanitizer

# Archive.org integration
try:
    from archive_org_collector import OptimizedArchiveOrgCollector as ArchiveOrgCollector
except ImportError:
    ArchiveOrgCollector = None

class HistoricalDataCollector:
    """Comprehensive system for collecting, processing, and cleaning historical data"""
    
    def __init__(self):
        self.setup_logging()
        self.data_sources_file = Path(__file__).parent / "data_sources.json"
        self.download_dir = config.london_historical_data / "downloads"
        self.processed_dir = config.london_historical_data / "processed"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize filename sanitizer
        self.filename_sanitizer = FilenameSanitizer(self.download_dir)
        
        # Statistics
        self.stats = {
            'sources_processed': 0,
            'files_downloaded': 0,
            'files_processed': 0,
            'files_cleaned': 0,
            'files_skipped': 0,
            'files_failed': 0,
            'files_sanitized': 0,
            'total_chars_before': 0,
            'total_chars_after': 0,
            'chars_removed': 0,
            'gutenberg_headers_removed': 0,
            'ocr_artifacts_fixed': 0,
            'html_markup_removed': 0,
            'duplicates_found': 0,
            'non_english_skipped': 0,
            'poor_quality_skipped': 0,
            'gutenberg_processed': 0,
            'gutenberg_accepted': 0,
            'encoding_issues_fixed': 0,
            'failed_downloads': [],
            'rejected_files': []  # New: Track rejected files with reasons
        }
        
        # Quality rejection logging
        self.rejection_log_file = config.london_historical_data / "logs" / f"rejected_files_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Deduplication tracking
        self.seen_hashes = set()
        
        # Load data sources
        self.data_sources = self.load_data_sources()
        
        # Initialize Archive.org collector if available
        self.archive_collector = None
        if ArchiveOrgCollector:
            try:
                self.archive_collector = ArchiveOrgCollector()
                self.logger.info("Archive.org API collector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Archive.org collector: {e}")
                self.archive_collector = None
    
    def setup_logging(self):
        """Setup logging"""
        # Create logs directory if it doesn't exist
        logs_dir = config.london_historical_data / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"historical_data_collector_{timestamp}.log"
        
        # Clear any existing handlers to avoid conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def load_data_sources(self) -> Dict[str, Any]:
        """Load data sources configuration"""
        if not self.data_sources_file.exists():
            self.logger.error(f"Data sources file not found: {self.data_sources_file}")
            return {}
        
        with open(self.data_sources_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        self.logger.info("Installing dependencies...")
        
        # Install from main requirements.txt
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        
        if requirements_file.exists():
            self.logger.info(f"Installing dependencies from {requirements_file}")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)], 
                             check=True, capture_output=True)
                self.logger.info("Dependencies installed successfully from requirements.txt")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install from requirements.txt: {e}")
                return False
        else:
            self.logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        # Check for system dependencies
        system_dependencies = {
            "pdftotext": "poppler-utils (install with: sudo apt-get install poppler-utils)"
        }
        
        missing_system_deps = []
        for dep, install_cmd in system_dependencies.items():
            try:
                # Check if system command exists
                result = subprocess.run(['which', dep], capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"✓ {dep} is available")
                else:
                    missing_system_deps.append((dep, install_cmd))
                    self.logger.warning(f"{dep} not available - {install_cmd}")
            except Exception as e:
                missing_system_deps.append((dep, install_cmd))
                self.logger.warning(f"{dep} not available - {install_cmd}")
        
        if missing_system_deps:
            self.logger.warning("Some system dependencies are missing:")
            for dep, install_cmd in missing_system_deps:
                self.logger.warning(f"  - {dep}: {install_cmd}")
            self.logger.warning("PDF processing may be limited, but other features will work.")
        else:
            self.logger.info("✓ All system dependencies are available")
        
        return True
    
    def convert_gutenberg_url(self, url: str) -> str:
        """Convert old Gutenberg URL format to new format"""
        # Convert from files/[ID]/[ID]-0.txt to ebooks/[ID].txt.utf-8
        if 'gutenberg.org/files/' in url and url.endswith('-0.txt'):
            # Extract the ID from the URL
            match = re.search(r'gutenberg\.org/files/(\d+)/\d+-0\.txt', url)
            if match:
                book_id = match.group(1)
                new_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
                self.logger.info(f"Converting Gutenberg URL: {url} -> {new_url}")
                return new_url
        return url

    def download_file(self, url: str, filename: str, source_name: str = None) -> bool:
        """Download a single file"""
        try:
            file_path = self.download_dir / filename
            
            # Skip if already exists
            if file_path.exists():
                self.logger.info(f"File already exists: {filename}")
                return True
            
            # Convert Gutenberg URLs to new format
            download_url = self.convert_gutenberg_url(url)
            
            self.logger.info(f"Downloading: {filename}")
            self.logger.info(f"URL: {download_url}")
            
            # Download with retry logic
            max_retries = 3
            last_error = None
            for attempt in range(max_retries):
                try:
                    urllib.request.urlretrieve(download_url, file_path)
                    self.stats['files_downloaded'] += 1
                    self.logger.info(f"Downloaded: {filename}")
                    
                    # Sanitize filename after download
                    self.sanitize_downloaded_file(file_path)
                    
                    return True
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Download attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(2)
                    else:
                        # Record failed download with details
                        failure_info = {
                            'source_name': source_name or 'Unknown',
                            'filename': filename,
                            'url': download_url,
                            'error': last_error,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.stats['failed_downloads'].append(failure_info)
                        self.logger.error(f"Failed to download {filename} after {max_retries} attempts: {e}")
                        return False
            
        except Exception as e:
            # Record failed download with details
            failure_info = {
                'source_name': source_name or 'Unknown',
                'filename': filename,
                'url': url,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.stats['failed_downloads'].append(failure_info)
            self.logger.error(f"Error downloading {filename}: {e}")
            return False
    
    def sanitize_downloaded_file(self, file_path: Path) -> bool:
        """Sanitize filename of a downloaded file"""
        try:
            original_name = file_path.name
            sanitized_name = self.filename_sanitizer.sanitize_filename(original_name)
            
            if original_name != sanitized_name:
                new_path = file_path.parent / sanitized_name
                file_path.rename(new_path)
                self.stats['files_sanitized'] += 1
                self.logger.info(f"Sanitized filename: {original_name} → {sanitized_name}")
                return True
            else:
                self.logger.debug(f"Filename already safe: {original_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to sanitize filename {file_path.name}: {e}")
            return False
    
    def sanitize_existing_filenames(self) -> int:
        """Sanitize all existing filenames in the download directory"""
        try:
            self.logger.info("Sanitizing existing filenames in download directory...")
            
            # Find all text files that might need sanitization
            text_files = list(self.download_dir.glob("*.txt"))
            pdf_files = list(self.download_dir.glob("*.pdf"))
            html_files = list(self.download_dir.glob("*.html")) + list(self.download_dir.glob("*.htm"))
            
            all_files = text_files + pdf_files + html_files
            sanitized_count = 0
            
            for file_path in all_files:
                original_name = file_path.name
                sanitized_name = self.filename_sanitizer.sanitize_filename(original_name)
                
                if original_name != sanitized_name:
                    new_path = file_path.parent / sanitized_name
                    try:
                        file_path.rename(new_path)
                        sanitized_count += 1
                        self.stats['files_sanitized'] += 1
                        self.logger.info(f"Sanitized: {original_name} → {sanitized_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to rename {original_name}: {e}")
            
            self.logger.info(f"Sanitized {sanitized_count} existing filenames")
            return sanitized_count
            
        except Exception as e:
            self.logger.error(f"Error during filename sanitization: {e}")
            return 0
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # Try system pdftotext first (if available)
        try:
            result = subprocess.run(['pdftotext', str(file_path), '-'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.logger.info(f"Successfully extracted text using system pdftotext: {file_path.name}")
                return result.stdout
        except FileNotFoundError:
            self.logger.warning(f"System pdftotext not found. Install with: sudo apt-get install poppler-utils")
        except Exception as e:
            self.logger.warning(f"System pdftotext failed for {file_path.name}: {e}")
        
        # Fallback to PyPDF2
        if PyPDF2:
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + '\n'
                self.logger.info(f"Successfully extracted text using PyPDF2: {file_path.name}")
                return text
            except Exception as e:
                self.logger.warning(f"PyPDF2 failed for {file_path.name}: {e}")
        
        self.logger.error(f"All PDF extraction methods failed for {file_path.name}")
        return ""
    
    def clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg headers and footers"""
        # Find start and end markers (flexible spacing)
        start_marker = r'\*\*\*\s*START\s+OF\s+THE\s+PROJECT\s+GUTENBERG.*?\*\*\*'
        end_marker = r'\*\*\*\s*END\s+OF\s+THE\s+PROJECT\s+GUTENBERG\s+EBOOK.*?\*\*\*'
        
        start_match = re.search(start_marker, text, re.DOTALL)
        end_match = re.search(end_marker, text, re.DOTALL)
        
        if start_match and end_match:
            text = text[start_match.end():end_match.start()]
            self.stats['gutenberg_headers_removed'] += 1
        elif start_match:
            text = text[start_match.end():]
            self.stats['gutenberg_headers_removed'] += 1
        elif end_match:
            text = text[:end_match.start()]
            self.stats['gutenberg_headers_removed'] += 1
        
        # Remove additional metadata
        metadata_patterns = [
            r'^(Title|Author|Release Date|Language|Character set encoding):.*$',
            r'^\[.*?\]$',
            r'^Produced by.*$',
            r'^End of Project Gutenberg.*$',
            r'^Project Gutenberg.*$',
            r'^This eBook is for.*$',
            r'^Copyright.*$',
            r'^Donations.*$'
        ]
        
        for pattern in metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()
    
    def clean_pdf_text(self, text: str) -> str:
        """Clean text extracted from PDFs"""
        # Remove page numbers
        text = re.sub(r'\[\s*Page\s*\d+\s*\]', '', text)
        text = re.sub(r'Page\s*\d+', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove library stamps
        text = re.sub(r'\b(Internet Archive|Google|Library of|Digitized by)\b.*?\n', '', text, flags=re.IGNORECASE)
        
        # Remove headers and footers
        text = re.sub(r'^[A-Z][A-Z\s]+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Chapter\s+[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common OCR errors
        ocr_corrections = {
            r'\b0\b': 'O',
            r'\b1\b': 'I',
            r'\b5\b': 'S',
            r'\b8\b': 'B',
            r'rn': 'm',
            r'cl': 'd',
            r'ii': 'n',
            r'vv': 'w',
            r'ſ': 's'
        }
        
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        self.stats['ocr_artifacts_fixed'] += 1
        return text.strip()
    
    def clean_html_text(self, html_content: str) -> str:
        """Extract clean text from HTML files"""
        if BeautifulSoup:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside']:
                for elem in soup.find_all(tag):
                    elem.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
        else:
            # Basic HTML tag removal
            text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Remove wiki metadata
        wiki_patterns = [
            r'This page was last edited on.*',
            r'Jump to navigation.*',
            r'Jump to search.*',
            r'From Wikisource.*',
            r'Retrieved from.*'
        ]
        
        for pattern in wiki_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        self.stats['html_markup_removed'] += 1
        return text.strip()
    
    def clean_xml_text(self, xml_content: str) -> str:
        """Extract clean text from XML files"""
        if BeautifulSoup:
            soup = BeautifulSoup(xml_content, 'xml')
            
            # Check if this is an Old Bailey XML file
            if self.is_old_bailey_xml(soup):
                return self.extract_old_bailey_text(soup)
            
            # Check if this is a London Lives XML file
            if self.is_london_lives_xml(soup):
                return self.extract_london_lives_text(soup)
            
            # Remove unwanted elements for general XML
            for tag in ['script', 'style', 'metadata', 'header', 'footer']:
                for elem in soup.find_all(tag):
                    elem.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
        else:
            # Basic XML tag removal
            text = re.sub(r'<[^>]+>', ' ', xml_content)
        
        # Remove XML-specific metadata
        xml_patterns = [
            r'<\?xml.*?\?>',
            r'<\!DOCTYPE.*?>',
            r'<metadata>.*?</metadata>',
            r'<header>.*?</header>',
            r'<footer>.*?</footer>'
        ]
        
        for pattern in xml_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        self.stats['html_markup_removed'] += 1  # Reuse HTML counter for XML
        return text.strip()
    
    def is_old_bailey_xml(self, soup) -> bool:
        """Check if this is an Old Bailey XML file"""
        # Look for Old Bailey specific elements
        tei_elements = soup.find_all('TEI.2')
        if tei_elements:
            # Check for Old Bailey specific div types
            sessions_paper = soup.find('div0', {'type': 'sessionsPaper'})
            ordinarys_account = soup.find('div0', {'type': 'ordinarysAccount'})
            return sessions_paper is not None or ordinarys_account is not None
        return False
    
    def is_london_lives_xml(self, soup) -> bool:
        """Check if this is a London Lives XML file"""
        # Look for London Lives specific elements
        # Check for name elements with London Lives ID pattern
        name_elements = soup.find_all('name')
        if name_elements:
            # Check if names have the London Lives ID pattern (nX-Y)
            for name in name_elements[:5]:  # Check first 5 names
                name_id = name.get('id', '')
                if name_id and name_id.startswith('n') and '-' in name_id:
                    return True
        
        # Check for London Lives specific elements
        geo_elements = soup.find_all('geo')
        occupation_elements = soup.find_all('occupation')
        date_elements = soup.find_all('date')
        
        # If we have multiple semantic markup elements, likely London Lives
        if len(geo_elements) > 0 and len(occupation_elements) > 0 and len(date_elements) > 0:
            return True
            
        return False
    
    def extract_old_bailey_text(self, soup) -> str:
        """Extract historical text from Old Bailey XML files using proper TEI structure"""
        extracted_text = []
        
        # Get session date from filename if available
        session_date = self.extract_session_date(soup)
        
        # Extract trial accounts (main narrative content)
        trial_accounts = soup.find_all('div1', {'type': 'trialAccount'})
        for trial in trial_accounts:
            trial_text = self.extract_trial_narrative(trial, session_date)
            if trial_text:
                extracted_text.append(trial_text)
        
        # Extract front matter (session information)
        front_matter = soup.find_all('div1', {'type': 'frontMatter'})
        for front in front_matter:
            front_text = self.extract_front_matter_narrative(front)
            if front_text:
                extracted_text.append(front_text)
        
        # Combine all extracted text
        combined_text = '\n\n'.join(extracted_text)
        
        # Clean up the text (minimal cleaning to preserve historical language)
        combined_text = self.clean_old_bailey_text(combined_text)
        
        self.stats['html_markup_removed'] += 1
        return combined_text.strip()
    
    def extract_session_date(self, soup) -> str:
        """Extract session date from XML"""
        # Try to get date from front matter
        date_elem = soup.find('date')
        if date_elem and date_elem.get('value'):
            return date_elem.get('value')
        
        # Fallback to filename parsing if available
        return None
    
    def extract_trial_narrative(self, trial_div, session_date) -> str:
        """Extract narrative text from a trial account"""
        trial_text = []
        
        # Extract trial title/heading
        head = trial_div.find('head')
        if head:
            trial_text.append(f"Trial: {head.get_text().strip()}")
        
        # Extract the main narrative text (preserve historical language)
        paragraphs = trial_div.find_all('p')
        for p in paragraphs:
            # Get text content while preserving historical spelling and punctuation
            p_text = self.extract_paragraph_narrative(p)
            if p_text.strip():
                trial_text.append(p_text)
        
        # Add session context if available
        if session_date:
            trial_text.insert(0, f"Session of {session_date}")
        
        return '\n'.join(trial_text)
    
    def extract_paragraph_narrative(self, paragraph) -> str:
        """Extract paragraph text preserving historical language"""
        # Get all text content, preserving the original structure
        text_parts = []
        
        for element in paragraph.descendants:
            if element.name is None:  # Text node
                text_parts.append(element.strip())
            elif element.name in ['persName', 'placeName', 'rs']:
                # Include person names, places, and other semantic elements as part of the narrative
                text_parts.append(element.get_text().strip())
        
        return ' '.join(text_parts)
    
    def extract_front_matter_narrative(self, front_div) -> str:
        """Extract narrative text from front matter"""
        front_text = []
        
        # Extract session information
        head = front_div.find('head')
        if head:
            front_text.append(f"Session: {head.get_text().strip()}")
        
        # Extract any narrative content
        paragraphs = front_div.find_all('p')
        for p in paragraphs:
            p_text = p.get_text().strip()
            if p_text and len(p_text) > 20:  # Only substantial content
                front_text.append(p_text)
        
        return '\n'.join(front_text)
    
    def extract_trial_content(self, trial_div) -> str:
        """Extract content from a trial account"""
        trial_text = []
        
        # Extract trial title/heading
        head = trial_div.find('head')
        if head:
            trial_text.append(f"TRIAL: {head.get_text().strip()}")
        
        # Extract all paragraphs
        paragraphs = trial_div.find_all('p')
        for p in paragraphs:
            # Extract person names with context
            p_text = self.extract_paragraph_with_names(p)
            if p_text.strip():
                trial_text.append(p_text)
        
        return '\n'.join(trial_text)
    
    def extract_paragraph_with_names(self, paragraph) -> str:
        """Extract paragraph text with person names properly formatted"""
        text_parts = []
        
        for element in paragraph.descendants:
            if element.name == 'persName':
                # Extract person name and attributes
                name_text = element.get_text().strip()
                name_type = element.get('type', '')
                name_id = element.get('id', '')
                
                # Format the name with context
                if name_type:
                    text_parts.append(f"[{name_type.upper()}: {name_text}]")
                else:
                    text_parts.append(name_text)
            elif element.name is None:  # Text node
                text_parts.append(element.strip())
        
        return ' '.join(text_parts)
    
    def extract_front_matter_content(self, front_div) -> str:
        """Extract content from front matter"""
        front_text = []
        
        # Extract title pages
        heads = front_div.find_all('head')
        for head in heads:
            front_text.append(f"TITLE: {head.get_text().strip()}")
        
        # Extract juror lists
        juror_lists = front_div.find_all('list')
        for juror_list in juror_lists:
            jurors = juror_list.find_all('item')
            if jurors:
                juror_names = [juror.get_text().strip() for juror in jurors]
                front_text.append(f"JURORS: {', '.join(juror_names)}")
        
        return '\n'.join(front_text)
    
    def extract_supplementary_content(self, supp_div) -> str:
        """Extract supplementary material content"""
        supp_text = []
        
        # Extract outcomes and verdicts
        paragraphs = supp_div.find_all('p')
        for p in paragraphs:
            p_text = p.get_text().strip()
            if p_text:
                supp_text.append(f"OUTCOME: {p_text}")
        
        return '\n'.join(supp_text)
    
    def extract_punishment_content(self, punishment_div) -> str:
        """Extract punishment summary content"""
        punishment_text = []
        
        # Extract sentences and punishments
        paragraphs = punishment_div.find_all('p')
        for p in paragraphs:
            p_text = p.get_text().strip()
            if p_text:
                punishment_text.append(f"SENTENCE: {p_text}")
        
        return '\n'.join(punishment_text)
    
    def clean_old_bailey_text(self, text: str) -> str:
        """Clean Old Bailey specific text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Clean up name tags that might have been missed
        text = re.sub(r'\[([A-Z_]+):\s*\]', '', text)  # Remove empty name tags
        
        # Remove common XML artifacts
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        
        return text.strip()
    
    def extract_london_lives_text(self, soup) -> str:
        """Extract historical text from London Lives XML files"""
        extracted_text = []
        
        # Extract document metadata
        doc_metadata = self.extract_london_lives_metadata(soup)
        if doc_metadata:
            extracted_text.append(doc_metadata)
        
        # Extract all paragraphs with semantic markup
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            p_text = self.extract_london_lives_paragraph(p)
            if p_text.strip():
                extracted_text.append(p_text)
        
        # Extract lists and other content
        lists = soup.find_all('list')
        for lst in lists:
            list_text = self.extract_london_lives_list(lst)
            if list_text.strip():
                extracted_text.append(list_text)
        
        # Combine all extracted text
        combined_text = '\n\n'.join(extracted_text)
        
        # Clean up the text
        combined_text = self.clean_london_lives_text(combined_text)
        
        self.stats['html_markup_removed'] += 1
        return combined_text.strip()
    
    def extract_london_lives_metadata(self, soup) -> str:
        """Extract minimal document metadata from London Lives XML"""
        metadata = []
        
        # Extract document title/heading - only if substantial
        heads = soup.find_all('head')
        for head in heads:
            head_text = head.get_text().strip()
            if head_text and len(head_text) > 10:  # Only substantial headings
                metadata.append(head_text)
        
        return '\n'.join(metadata)
    
    def extract_london_lives_paragraph(self, paragraph) -> str:
        """Extract paragraph text with minimal markup for better narrative flow"""
        text_parts = []
        
        for element in paragraph.descendants:
            if element.name == 'name':
                # Extract person name - just the name, no verbose attributes
                name_text = element.get_text().strip()
                text_parts.append(name_text)
                    
            elif element.name == 'geo':
                # Extract geographic information - just the place name
                geo_text = element.get_text().strip()
                text_parts.append(geo_text)
                    
            elif element.name == 'occupation':
                # Extract occupation - just the occupation
                occ_text = element.get_text().strip()
                text_parts.append(occ_text)
                
            elif element.name == 'date':
                # Extract date information - just the date
                date_text = element.get_text().strip()
                text_parts.append(date_text)
                    
            elif element.name is None:  # Text node
                text_parts.append(element.strip())
        
        return ' '.join(text_parts)
    
    def extract_london_lives_list(self, list_elem) -> str:
        """Extract list content with proper formatting"""
        list_items = []
        
        items = list_elem.find_all('item')
        for item in items:
            item_text = self.extract_london_lives_paragraph(item)
            if item_text.strip():
                list_items.append(f"• {item_text}")
        
        return '\n'.join(list_items)
    
    def clean_london_lives_text(self, text: str) -> str:
        """Clean London Lives specific text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Clean up markup artifacts
        text = re.sub(r'\[([A-Z_]+):\s*\]', '', text)  # Remove empty markup tags
        text = re.sub(r'\[([A-Z_]+):\s*\(\s*\)\]', r'[\1]', text)  # Remove empty attributes
        
        # Remove common XML artifacts
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        
        return text.strip()
    
    def collect_archive_org_data(self, max_items: int = 500) -> bool:
        """
        Collect data using Archive.org API (enhanced method)
        
        Args:
            max_items: Maximum number of items to collect
        
        Returns:
            True if collection successful
        """
        if not self.archive_collector:
            self.logger.warning("Archive.org collector not available, skipping API collection")
            return False
        
        self.logger.info("Starting Archive.org API data collection...")
        
        try:
            # Use the Archive.org collector
            success = self.archive_collector.collect_london_sources(max_items)
            
            if success:
                # Process the downloaded files
                archive_dir = self.archive_collector.download_dir
                self.logger.info(f"Processing downloaded files from: {archive_dir}")
                
                processed_count = 0
                for file_path in archive_dir.glob("*.txt"):
                    if self.process_file(file_path):
                        processed_count += 1
                
                self.logger.info(f"Processed {processed_count} Archive.org files")
                return True
            else:
                self.logger.error("Archive.org collection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Archive.org collection error: {e}")
            return False
    
    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization with repetitive pattern cleaning"""
        # Fix basic encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '—')
        text = text.replace('â€"', '–')
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove repetitive patterns that cause model issues
        text = self.clean_repetitive_patterns(text)
        
        # Basic cleaning - remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def clean_repetitive_patterns(self, text: str) -> str:
        """Remove repetitive patterns that can cause model generation issues"""
        # Remove repetitive bullet points (5+ in a row)
        text = re.sub(r'•\s*•\s*•\s*•\s*•+', '', text)
        
        # Remove repetitive carets (5+ in a row)  
        text = re.sub(r'\^\s*\^\s*\^\s*\^\s*\^+', '', text)
        
        # Remove any character repeated 5+ times
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove lines that are just repetitive characters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Check if line is mostly repetitive characters
                if len(line) > 10:
                    # Count unique characters
                    unique_chars = len(set(line))
                    if unique_chars < 3:  # Less than 3 unique characters in a long line
                        continue  # Skip this line
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_existing_processed_data(self) -> bool:
        """Clean repetitive patterns from existing processed data files"""
        try:
            self.logger.info("🧹 Cleaning existing processed data...")
            
            # Find all processed files
            processed_files = list(self.processed_dir.glob("cleaned_*.txt"))
            if not processed_files:
                self.logger.warning("No processed files found to clean")
                return False
            
            self.logger.info(f"Found {len(processed_files)} processed files to clean")
            
            cleaned_count = 0
            total_chars_before = 0
            total_chars_after = 0
            
            for file_path in processed_files:
                try:
                    # Read original file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    total_chars_before += len(original_content)
                    
                    # Clean the content
                    cleaned_content = self.clean_repetitive_patterns(original_content)
                    
                    # Only write if content changed
                    if cleaned_content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        
                        chars_removed = len(original_content) - len(cleaned_content)
                        self.logger.info(f"Cleaned {file_path.name}: removed {chars_removed:,} characters")
                        cleaned_count += 1
                    
                    total_chars_after += len(cleaned_content)
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning {file_path.name}: {e}")
                    continue
            
            # Clean the comprehensive corpus
            corpus_file = config.london_historical_data / "london_historical_corpus_comprehensive.txt"
            if corpus_file.exists():
                self.logger.info("Cleaning comprehensive corpus...")
                try:
                    with open(corpus_file, 'r', encoding='utf-8') as f:
                        corpus_content = f.read()
                    
                    cleaned_corpus = self.clean_repetitive_patterns(corpus_content)
                    
                    if cleaned_corpus != corpus_content:
                        with open(corpus_file, 'w', encoding='utf-8') as f:
                            f.write(cleaned_corpus)
                        
                        corpus_chars_removed = len(corpus_content) - len(cleaned_corpus)
                        self.logger.info(f"Cleaned corpus: removed {corpus_chars_removed:,} characters")
                        cleaned_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning corpus: {e}")
            
            # Print summary
            chars_removed = total_chars_before - total_chars_after
            self.logger.info(f"✅ Cleaning completed:")
            self.logger.info(f"   Files cleaned: {cleaned_count}")
            self.logger.info(f"   Characters removed: {chars_removed:,}")
            self.logger.info(f"   Original size: {total_chars_before:,}")
            self.logger.info(f"   Cleaned size: {total_chars_after:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {e}")
            return False
    
    def is_obviously_structured_line(self, line: str) -> bool:
        """Check if a line is obviously structured data (less aggressive)"""
        # Only filter the most obvious structured data
        obvious_indicators = [
            r'^\d+\s*$',                    # Just numbers
            r'^[A-Z\s]{10,}$',             # All caps (long lines)
            r'^\w+\s*:\s*\w+\s*$',         # Simple key: value format
            r'\[[A-Z_]+\s*:\s*[^\]]+\]',   # [TYPE: value] tags
        ]
        
        for pattern in obvious_indicators:
            if re.search(pattern, line):
                return True
        
        return False
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
        content_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
        
        if content_hash in self.seen_hashes:
            self.stats['duplicates_found'] += 1
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text"""
        if not detect:
            return None
        
        try:
            sample = text[:1000] if len(text) > 1000 else text
            return detect(sample)
        except LangDetectException:
            return None
    
    def is_non_english_filename(self, filename: str) -> bool:
        """Check if filename contains non-English characters (Arabic, Chinese, etc.)"""
        # Check for Arabic, Chinese, Japanese, Korean, Cyrillic characters
        non_english_ranges = [
            '\u0600-\u06FF',  # Arabic
            '\u4E00-\u9FFF',  # Chinese/Japanese/Korean
            '\u0400-\u04FF',  # Cyrillic
            '\u0590-\u05FF',  # Hebrew
            '\u0900-\u097F',  # Devanagari
        ]
        
        for char_range in non_english_ranges:
            if re.search(f'[{char_range}]', filename):
                return True
        
        return False
    
    def is_project_gutenberg_text(self, text: str, filename: str = "") -> bool:
        """Check if text is from Project Gutenberg"""
        # Check filename patterns first
        if filename:
            filename_lower = filename.lower()
            # Gutenberg-specific patterns
            gutenberg_patterns = [
                'project_gutenberg', 'gutenberg_', '_gutenberg', 'pg_', '_pg_'
            ]
            if any(pattern in filename_lower for pattern in gutenberg_patterns):
                return True
            
            # Numeric patterns that suggest Gutenberg (but could be other sources)
            if (re.search(r'_?pg\d+', filename_lower) or 
                re.search(r'_\d{4,}_txt', filename_lower) or  # 4+ digit numbers like _31412_txt
                re.search(r'_\d+\.txt', filename_lower)):     # Numbers before .txt
                return True
        
        # Check content markers
        gutenberg_markers = [
            'PROJECT GUTENBERG',
            'gutenberg.org',
            'START OF THE PROJECT GUTENBERG',
            'END OF THE PROJECT GUTENBERG',
            'This eBook is for the use of anyone',
            'www.gutenberg.org',
            'Project Gutenberg-tm',
            'Project Gutenberg Literary Archive Foundation'
        ]
        
        text_upper = text.upper()
        return any(marker.upper() in text_upper for marker in gutenberg_markers)
    
    def validate_gutenberg_quality(self, text: str) -> bool:
        """Validate Project Gutenberg text quality with relaxed criteria"""
        # Remove Gutenberg headers/footers for quality assessment
        cleaned_text = self.clean_gutenberg_text(text)
        
        # More lenient criteria for Gutenberg texts
        if len(cleaned_text) < 500:  # Relaxed from 200
            return False
        
        words = cleaned_text.split()
        if len(words) < 100:  # Relaxed from 50
            return False
        
        # Check for reasonable English word ratio (more lenient)
        meaningful_words = 0
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 2 and sum(c.isalpha() for c in clean_word) / len(clean_word) > 0.6:
                meaningful_words += 1
        
        meaningful_ratio = meaningful_words / len(words)
        return meaningful_ratio >= 0.4  # More lenient: 40% vs 50%
    
    def is_duplicate(self, text: str) -> bool:
        """Check if this text content has already been processed"""
        # Use first 1000 characters for duplicate detection
        text_sample = text[:1000] if len(text) > 1000 else text
        text_hash = hashlib.sha256(text_sample.encode('utf-8')).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def is_historical_text_file(self, filename: str) -> bool:
        """Check if file appears to be historical text content from any source"""
        filename_lower = filename.lower()
        
        # Historical text file indicators
        historical_indicators = [
            # Project Gutenberg
            'project_gutenberg', 'gutenberg', 'pg',
            # Archives and libraries
            'london', 'british', 'archive', 'library', 'manuscript',
            # Historical content types
            'diary', 'memoir', 'chronicle', 'journal', 'correspondence',
            'proceedings', 'records', 'register', 'survey', 'history',
            # Time periods
            'medieval', 'tudor', 'stuart', 'georgian', 'regency', 'victorian',
            # Geographic
            'westminster', 'southwark', 'cheapside', 'newgate', 'tower',
            # Document types
            'vol', 'volume', 'part', 'book', 'text', 'work'
        ]
        
        # Check for historical indicators
        for indicator in historical_indicators:
            if indicator in filename_lower:
                return True
        
        # Check for numeric patterns suggesting catalogued historical texts
        if (re.search(r'_\d{3,}', filename_lower) or     # 3+ digit catalog numbers
            re.search(r'\d{4}', filename_lower)):        # Years (1600-2099)
            return True
        
        return False
    
    def normalize_text(self, text: str) -> str:
        """Normalize text and break up single long lines for training compatibility"""
        if not text:
            return text
        
        # Handle single long lines (common in downloaded files)
        lines = text.split('\n')
        if len(lines) == 1 and len(text) > 1000:
            # Single line that's very long - need to break it up
            text = self.break_long_line(text)
        
        # Basic normalization
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\r', '\n', text)    # Handle old Mac line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize whitespace
        
        return text.strip()
    
    def break_long_line(self, text: str, max_line_length: int = 80) -> str:
        """Break a single long line into reasonable chunks for training"""
        if '\n' in text:
            return text  # Already has line breaks
        
        # Try to break at sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            # Recombine into reasonable line lengths
            lines = []
            current_line = ""
            
            for sentence in sentences:
                if len(current_line) + len(sentence) + 1 <= max_line_length * 2:  # Allow longer lines for readability
                    if current_line:
                        current_line += " " + sentence
                    else:
                        current_line = sentence
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = sentence
            
            if current_line:
                lines.append(current_line)
            
            return '\n'.join(lines)
        
        # Fallback: break at word boundaries
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return '\n'.join(lines)
    
    def clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text and handle formatting"""
        if not text:
            return text
        
        # Remove common Gutenberg headers and footers
        gutenberg_patterns = [
            r'^\*\*\* START OF .*?\*\*\*.*?$',
            r'^\*\*\* END OF .*?\*\*\*.*?$',
            r'^The Project Gutenberg eBook.*?$',
            r'^This eBook is for the use of.*?$',
            r'^You may copy it.*?$',
            r'^Title:.*?$',
            r'^Author:.*?$',
            r'^Release date:.*?$',
            r'^Language:.*?$',
            r'^Credits:.*?$',
            r'^Produced by.*?$',
        ]
        
        for pattern in gutenberg_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple blank lines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        
        return text.strip()
    
    def log_rejected_file(self, file_path: Path, reason: str, details: dict = None):
        """Log rejected file with detailed reason for later review"""
        try:
            rejection_entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'file_path': str(file_path),
                'filename': file_path.name,
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'rejection_reason': reason,
                'details': details or {},
            }
            
            # Add file preview for manual review
            if file_path.exists() and file_path.stat().st_size < 10000:  # Only for small files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        preview = f.read(500)  # First 500 chars
                        rejection_entry['preview'] = preview
                except:
                    rejection_entry['preview'] = "Could not read file preview"
            
            self.stats['rejected_files'].append(rejection_entry)
            
            # Also log to console for immediate visibility
            self.logger.info(f"REJECTED: {file_path.name} - {reason}")
            if details:
                for key, value in details.items():
                    self.logger.info(f"   {key}: {value}")
                    
        except Exception as e:
            self.logger.error(f"Failed to log rejection for {file_path}: {e}")
    
    def save_rejection_log(self):
        """Save rejection log to JSON file for offline review"""
        try:
            # Ensure logs directory exists
            self.rejection_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive rejection report
            rejection_report = {
                'collection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_files_processed': self.stats['files_processed'],
                'total_rejected': len(self.stats['rejected_files']),
                'rejection_summary': self.generate_rejection_summary(),
                'rejected_files': self.stats['rejected_files']
            }
            
            with open(self.rejection_log_file, 'w', encoding='utf-8') as f:
                json.dump(rejection_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Rejection log saved: {self.rejection_log_file}")
            self.logger.info(f"   Total rejected: {len(self.stats['rejected_files'])}")
            
        except Exception as e:
            self.logger.error(f"Failed to save rejection log: {e}")
    
    def generate_rejection_summary(self) -> dict:
        """Generate summary statistics for rejection reasons"""
        summary = {}
        for rejected in self.stats['rejected_files']:
            reason = rejected['rejection_reason']
            summary[reason] = summary.get(reason, 0) + 1
        return summary
    
    def analyze_text_quality(self, text: str, filename: str = "") -> dict:
        """Analyze text quality with detailed reporting for rejection logging"""
        analysis = {
            'is_good_quality': True,
            'details': {
                'text_length': len(text),
                'line_count': len(text.split('\n')),
                'word_count': len(text.split()),
                'is_project_gutenberg': False,
                'ocr_issues': [],
                'advertisement_indicators': [],
                'meaningful_word_ratio': 0.0,
                'rejection_reasons': []
            }
        }
        
        # Basic length checks
        if len(text) < 200:
            analysis['is_good_quality'] = False
            analysis['details']['rejection_reasons'].append('Text too short (< 200 chars)')
            return analysis
        
        # Special handling for historical texts (check BEFORE line count)
        is_gutenberg = self.is_project_gutenberg_text(text, filename)
        is_historical = self.is_historical_text_file(filename)
        
        if is_gutenberg:
            analysis['details']['is_project_gutenberg'] = True
            self.stats['gutenberg_processed'] += 1
            result = self.validate_gutenberg_quality(text)
            if result:
                self.stats['gutenberg_accepted'] += 1
                analysis['details']['rejection_reasons'].append('Project Gutenberg - accepted with relaxed criteria')
                return analysis
            else:
                analysis['is_good_quality'] = False
                analysis['details']['rejection_reasons'].append('Project Gutenberg - failed relaxed quality check')
            return analysis
        elif is_historical:
            # Apply relaxed criteria for other historical texts too
            analysis['details']['is_historical_text'] = True
            if len(text) >= 1000 and len(text.split()) >= 100:  # Very relaxed for historical content
                analysis['details']['rejection_reasons'].append('Historical text - accepted with relaxed criteria')
                return analysis
        
        lines = text.split('\n')
        if len(lines) < 5:
            analysis['is_good_quality'] = False
            analysis['details']['rejection_reasons'].append('Too few lines (< 5)')
            return analysis
        
        # Check for poor OCR indicators
        poor_ocr_patterns = {
            'long_capitals': r'[A-Z]{5,}\s+[A-Z]{5,}',
            'spaced_letters': r'\b[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\b',
            'special_chars': r'[!@#$%^&*()]{3,}',
            'mixed_nums_letters': r'\b\d+[A-Za-z]+\d+\b',
            'long_nonword_sequences': r'[^\w\s]{10,}'
        }
        
        for pattern_name, pattern in poor_ocr_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                analysis['details']['ocr_issues'].append({
                    'type': pattern_name,
                    'count': len(matches),
                    'examples': matches[:3]  # First 3 examples
                })
        
        ocr_issue_count = len(analysis['details']['ocr_issues'])
        if ocr_issue_count >= 3:
            analysis['is_good_quality'] = False
            analysis['details']['rejection_reasons'].append(f'Too many OCR artifacts ({ocr_issue_count} types)')
        
        # Check for advertisement-heavy content
        ad_patterns = {
            'this_day_published': r'this day is published',
            'just_ready': r'just ready',
            'elegantly_bound': r'elegantly bound',
            'price_notation': r'price \d+s',
            'paternoster_row': r'paternoster.?row',
            'corner_of': r'corner of',
            'publishers': r'publishers?[,:]',
            'now_ready': r'now ready',
            'new_novels': r'new novels?',
            'advertisements': r'advertisements?',
            'for_sale': r'for sale',
            'to_be_had': r'to be had'
        }
        
        text_lower = text.lower()
        for pattern_name, pattern in ad_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                analysis['details']['advertisement_indicators'].append({
                    'type': pattern_name,
                    'count': len(matches)
                })
        
        ad_density = len(analysis['details']['advertisement_indicators']) / max(len(lines), 1)
        if ad_density > 0.3:
            analysis['is_good_quality'] = False
            analysis['details']['rejection_reasons'].append(f'High advertisement density ({ad_density:.2f})')
        
        # Check ratio of meaningful words to total words
        words = text.split()
        if len(words) < 50:
            analysis['is_good_quality'] = False
            analysis['details']['rejection_reasons'].append('Too few words (< 50)')
            return analysis
        
        # Count words that look like real English words
        meaningful_words = 0
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 3 and sum(c.isalpha() for c in clean_word) / len(clean_word) > 0.7:
                meaningful_words += 1
        
        meaningful_ratio = meaningful_words / len(words)
        analysis['details']['meaningful_word_ratio'] = meaningful_ratio
        
        if meaningful_ratio < 0.5:
            analysis['is_good_quality'] = False
            analysis['details']['rejection_reasons'].append(f'Low meaningful word ratio ({meaningful_ratio:.2f} < 0.5)')
        
        return analysis
    
    def is_good_quality_text(self, text: str) -> bool:
        """Check if text is good quality (backward compatibility wrapper)"""
        return self.analyze_text_quality(text)['is_good_quality']
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file"""
        try:
            # Quick filename-based language check for obviously non-English files
            filename = file_path.name
            if self.is_non_english_filename(filename):
                self.log_rejected_file(file_path, "Non-English filename", {
                    "detected_script": "Non-Latin characters in filename"
                })
                self.stats['non_english_skipped'] += 1
                self.stats['files_skipped'] += 1
                return True
                
            self.logger.info(f"Processing: {file_path.name}")
            
            # Read file based on type
            # Check for text files (.txt, .txt.utf-8, _txt.utf-8, etc.)
            is_text_file = (
                file_path.suffix.lower() == '.txt' or
                (file_path.suffix.lower() in ['.utf-8', '.utf8'] and 
                 ('txt' in file_path.name.lower() or 
                  file_path.stem.lower().endswith('_txt') or
                  '_txt.' in file_path.name.lower()))
            )
            
            if is_text_file:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                cleaned_text = self.clean_gutenberg_text(text)
                
            elif file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(file_path)
                if not text:
                    self.logger.error(f"Could not extract text from PDF: {file_path.name}")
                    return False
                cleaned_text = self.clean_pdf_text(text)
                
                
            elif file_path.suffix.lower() in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                cleaned_text = self.clean_html_text(html_content)
                
            elif file_path.suffix.lower() == '.xml':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    xml_content = f.read()
                cleaned_text = self.clean_xml_text(xml_content)
                
            else:
                # Check if file without extension is HTML or text content
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check if it looks like HTML
                    if '<html' in content.lower() or '<!doctype' in content.lower():
                        self.logger.info(f"Detected HTML content in file without extension: {file_path.name}")
                        cleaned_text = self.clean_html_text(content)
                    # Check if it looks like text (contains readable text, not just binary)
                    elif len(content) > 100 and any(c.isalpha() for c in content[:1000]):
                        self.logger.info(f"Detected text content in file without extension: {file_path.name}")
                        cleaned_text = self.clean_gutenberg_text(content)
                    else:
                        self.logger.warning(f"Unsupported file type: {file_path.name}")
                        return False
                except Exception as e:
                    self.logger.warning(f"Could not read file {file_path.name}: {e}")
                    return False
            
            # Normalize text
            cleaned_text = self.normalize_text(cleaned_text)
            
            # Check for duplicates
            if self.is_duplicate(cleaned_text):
                self.log_rejected_file(file_path, "Duplicate content", {
                    "content_hash": hashlib.sha256(cleaned_text[:1000].encode()).hexdigest()[:16]
                })
                self.stats['files_skipped'] += 1
                return True
            
            # Language detection - skip obviously non-English content
            if detect:
                language = self.detect_language(cleaned_text)
                if language and language != 'en':
                    self.log_rejected_file(file_path, "Non-English content", {
                        "detected_language": language,
                        "confidence": "langdetect library"
                    })
                    self.stats['non_english_skipped'] += 1
                    self.stats['files_skipped'] += 1
                    return True
            
            # Quality check - skip poor OCR and advertisement-heavy content
            quality_result = self.analyze_text_quality(cleaned_text, file_path.name)
            if not quality_result['is_good_quality']:
                self.log_rejected_file(file_path, "Poor content quality", quality_result['details'])
                self.stats['poor_quality_skipped'] += 1
                self.stats['files_skipped'] += 1
                return True
            
            # Update statistics
            self.stats['total_chars_before'] += len(text) if 'text' in locals() else 0
            self.stats['total_chars_after'] += len(cleaned_text)
            self.stats['chars_removed'] += (len(text) if 'text' in locals() else 0) - len(cleaned_text)
            
            # Save cleaned text to processed directory
            cleaned_file = self.processed_dir / f"cleaned_{file_path.stem}.txt"
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            self.stats['files_processed'] += 1
            self.stats['files_cleaned'] += 1
            self.logger.info(f"Successfully processed: {file_path.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            self.stats['files_failed'] += 1
            return False
    
    def download_and_process_sources(self, max_sources: Optional[int] = None) -> bool:
        """Download and process all data sources"""
        self.logger.info("Starting unified data collection and processing")
        
        # Show data source configuration
        slm_config = config.slm_config
        self.logger.info("📊 Data source configuration:")
        self.logger.info(f"   Old Bailey: {'✅ ENABLED' if slm_config.get('enable_old_bailey', True) else '❌ DISABLED'}")
        self.logger.info(f"   London Lives: {'✅ ENABLED' if slm_config.get('enable_london_lives', False) else '❌ DISABLED'}")
        self.logger.info(f"   Literature: {'✅ ENABLED' if slm_config.get('enable_literature', False) else '❌ DISABLED'}")
        self.logger.info(f"   Newspapers: {'✅ ENABLED' if slm_config.get('enable_newspapers', False) else '❌ DISABLED'}")
        self.logger.info(f"   Diaries: {'✅ ENABLED' if slm_config.get('enable_diaries', False) else '❌ DISABLED'}")
        self.logger.info(f"   Archive.org: {'✅ ENABLED' if slm_config.get('enable_archive_org', True) else '❌ DISABLED'}")
        self.logger.info(f"   Aggressive cleaning: {'✅ ENABLED' if slm_config.get('aggressive_cleaning', False) else '❌ DISABLED'}")
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Sanitize existing filenames first
        self.logger.info("Sanitizing existing filenames...")
        self.sanitize_existing_filenames()
        
        # Process sources based on configuration flags
        sources_processed = 0
        for source_id, source_data in self.data_sources.get('historical_sources', {}).items():
            if max_sources and sources_processed >= max_sources:
                break
            
            # Check configuration flags for data sources
            slm_config = config.slm_config
            
            # Skip sources based on configuration
            if source_id == 'old_bailey' and not slm_config.get('enable_old_bailey', True):
                self.logger.info(f"🚫 SKIPPING Old Bailey data (disabled in config): {source_data.get('name', source_id)}")
                continue
            elif source_id == 'london_lives' and not slm_config.get('enable_london_lives', False):
                self.logger.info(f"🚫 SKIPPING London Lives data (disabled in config): {source_data.get('name', source_id)}")
                continue
            elif source_id in ['parish_records', 'criminal_justice']:
                self.logger.info(f"🚫 SKIPPING structured data source: {source_data.get('name', source_id)}")
                continue
            elif source_data.get('type') == 'literature' and not slm_config.get('enable_literature', False):
                self.logger.info(f"🚫 SKIPPING Literature data (disabled in config): {source_data.get('name', source_id)}")
                continue
            elif source_data.get('type') == 'newspaper' and not slm_config.get('enable_newspapers', False):
                self.logger.info(f"🚫 SKIPPING Newspaper data (disabled in config): {source_data.get('name', source_id)}")
                continue
            elif source_data.get('type') == 'diary' and not slm_config.get('enable_diaries', False):
                self.logger.info(f"🚫 SKIPPING Diary data (disabled in config): {source_data.get('name', source_id)}")
                continue
            elif 'archive.org' in source_data.get('url', '') and not slm_config.get('enable_archive_org', True):
                self.logger.info(f"🚫 SKIPPING Archive.org source (disabled in config): {source_data.get('name', source_id)}")
                continue
            
            self.logger.info(f"Processing data source: {source_data.get('name', source_id)}")
            
            # Download files if URL provided
            if 'download_url' in source_data:
                url = source_data['download_url']
                filename = f"{source_id}_{Path(urlparse(url).path).name}"
                
                if self.download_file(url, filename, source_data.get('name', source_id)):
                    # Check for sanitized filename after download
                    sanitized_filename = self.filename_sanitizer.sanitize_filename(filename)
                    file_path = self.download_dir / sanitized_filename
                    if file_path.exists():
                        self.process_file(file_path)
            
            # Process any existing files for this source
            for file_path in self.download_dir.glob(f"{source_id}_*"):
                if file_path.is_file():
                    self.process_file(file_path)
            
            # Process nested sources if they exist
            if 'sources' in source_data and isinstance(source_data['sources'], list):
                self.logger.info(f"Processing {len(source_data['sources'])} nested sources for {source_data.get('name', source_id)}")
                for nested_source in source_data['sources']:
                    if max_sources and sources_processed >= max_sources:
                        break
                    
                    # Check if this nested source type is enabled
                    nested_type = nested_source.get('type', 'unknown')
                    if nested_type == 'literature' and not slm_config.get('enable_literature', False):
                        self.logger.info(f"🚫 SKIPPING nested Literature: {nested_source.get('name', 'unnamed')}")
                        continue
                    elif nested_type == 'newspaper' and not slm_config.get('enable_newspapers', False):
                        self.logger.info(f"🚫 SKIPPING nested Newspaper: {nested_source.get('name', 'unnamed')}")
                        continue
                    elif nested_type == 'diary' and not slm_config.get('enable_diaries', False):
                        self.logger.info(f"🚫 SKIPPING nested Diary: {nested_source.get('name', 'unnamed')}")
                        continue
                    elif 'archive.org' in nested_source.get('url', '') and not slm_config.get('enable_archive_org', True):
                        self.logger.info(f"🚫 SKIPPING nested Archive.org source: {nested_source.get('name', 'unnamed')}")
                        continue
                    
                    # Download nested source if it has a download_url
                    if 'download_url' in nested_source:
                        url = nested_source['download_url']
                        filename = f"{source_id}_{nested_source.get('gutenberg_id', nested_source.get('name', 'unnamed')).replace(' ', '_')}_{Path(urlparse(url).path).name}"
                        
                        self.logger.info(f"Downloading nested source: {nested_source.get('name', 'unnamed')}")
                        if self.download_file(url, filename, nested_source.get('name', 'unnamed')):
                            # Check for sanitized filename after download
                            sanitized_filename = self.filename_sanitizer.sanitize_filename(filename)
                            file_path = self.download_dir / sanitized_filename
                            if file_path.exists():
                                self.process_file(file_path)
                                self.logger.info(f"✅ Successfully processed nested source: {nested_source.get('name', 'unnamed')}")
                            else:
                                self.logger.warning(f"⚠️ Downloaded file not found: {sanitized_filename}")
                        else:
                            self.logger.warning(f"⚠️ Failed to download nested source: {nested_source.get('name', 'unnamed')}")
                    else:
                        self.logger.info(f"ℹ️ No download_url for nested source: {nested_source.get('name', 'unnamed')}")
            
            sources_processed += 1
            self.stats['sources_processed'] += 1
        
        # Collect additional data using Archive.org API if enabled
        if slm_config.get('enable_archive_org', True) and self.archive_collector:
            self.logger.info("🔍 Collecting additional data from Archive.org API...")
            self.collect_archive_org_data(max_items=500)
        
        # Process ALL remaining files in the directory (including HTML without extensions)
        self.logger.info("Processing all remaining files in download directory...")
        all_files = list(self.download_dir.glob("*"))
        for file_path in all_files:
            if (file_path.is_file() and 
                not file_path.name.startswith('cleaned_') and 
                not file_path.name.endswith('.json') and
                not file_path.name.endswith('.log') and
                file_path.name != 'london_historical_corpus_comprehensive.txt'):
                self.process_file(file_path)
        
        # Process files in manual_downloads subfolders based on configuration
        self.logger.info("Processing manual downloads in subfolders...")
        manual_downloads_dir = self.download_dir / "manual_downloads"
        if manual_downloads_dir.exists():
            # Get configuration flags
            slm_config = config.slm_config
            
            # Process all files in subfolders recursively based on configuration
            for file_path in manual_downloads_dir.rglob("*"):
                if (file_path.is_file() and 
                    not file_path.name.startswith('cleaned_') and 
                    not file_path.name.endswith('.json') and
                    not file_path.name.endswith('.log')):
                    
                    # Check if this is London Lives data and if it's enabled
                    if 'london_lives' in str(file_path).lower():
                        if slm_config.get('enable_london_lives', False):
                            self.logger.info(f"Processing London Lives file: {file_path.relative_to(self.download_dir)}")
                            self.process_file(file_path)
                        else:
                            self.logger.info(f"🚫 SKIPPING London Lives file (disabled in config): {file_path.relative_to(self.download_dir)}")
                    elif 'old_bailey' in str(file_path).lower():
                        # For Old Bailey, ONLY process XML files, skip everything else
                        if file_path.suffix.lower() == '.xml':
                            if slm_config.get('enable_old_bailey', True):
                                self.logger.info(f"Processing Old Bailey XML file: {file_path.relative_to(self.download_dir)}")
                                self.process_file(file_path)
                            else:
                                self.logger.info(f"🚫 SKIPPING Old Bailey file (disabled in config): {file_path.relative_to(self.download_dir)}")
                        else:
                            self.logger.info(f"🚫 SKIPPING Old Bailey non-XML file: {file_path.relative_to(self.download_dir)} (ONLY XML files are processed)")
                            continue  # Skip to next file, don't process anything else
                    else:
                        # Process other manual downloads
                        self.logger.info(f"Processing manual download: {file_path.relative_to(self.download_dir)}")
                        self.process_file(file_path)
        
        # Create comprehensive corpus
        self.create_comprehensive_corpus()
        
        # Save statistics
        self.save_statistics()
        
        # Save detailed rejection log for quality review
        self.save_rejection_log()
        
        return True
    
    def create_comprehensive_corpus(self):
        """Create comprehensive corpus from all cleaned files with proper text segmentation"""
        self.logger.info("Creating comprehensive corpus with proper text segmentation")
        
        corpus_file = config.london_historical_data / "london_historical_corpus_comprehensive.txt"
        
        cleaned_files = list(self.processed_dir.glob("cleaned_*.txt"))
        total_chars = 0
        total_segments = 0
        
        with open(corpus_file, 'w', encoding='utf-8') as corpus_f:
            for cleaned_file in cleaned_files:
                try:
                    with open(cleaned_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content and len(content) > 100:
                            # Split content into proper segments
                            segments = self.split_into_training_segments(content)
                            
                            for segment in segments:
                                if len(segment.strip()) > 50:  # Only segments with substantial content
                                    corpus_f.write(segment.strip())
                                    corpus_f.write('\n\n')
                                    total_chars += len(segment)
                                    total_segments += 1
                            
                            self.logger.info(f"Processed {cleaned_file.name}: {len(segments)} segments")
                            
                except Exception as e:
                    self.logger.error(f"Error reading {cleaned_file}: {e}")
        
        self.logger.info(f"Created comprehensive corpus: {corpus_file}")
        self.logger.info(f"Total characters: {total_chars:,}")
        self.logger.info(f"Total segments: {total_segments:,}")
        self.logger.info(f"Average segment length: {total_chars // max(total_segments, 1):,} characters")
    
    def split_into_training_segments(self, text: str, max_length: int = 2000, min_length: int = 100) -> List[str]:
        """
        Split text into training-friendly segments
        
        Args:
            text: Input text to split
            max_length: Maximum characters per segment
            min_length: Minimum characters per segment
        
        Returns:
            List of text segments
        """
        segments = []
        
        # First, try to split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_segment = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed max_length, save current segment
            if current_segment and len(current_segment) + len(paragraph) + 2 > max_length:
                if len(current_segment) >= min_length:
                    segments.append(current_segment.strip())
                current_segment = paragraph
            else:
                if current_segment:
                    current_segment += "\n\n" + paragraph
                else:
                    current_segment = paragraph
        
        # Add the last segment
        if current_segment and len(current_segment) >= min_length:
            segments.append(current_segment.strip())
        
        # If we still have very long segments, split them further
        final_segments = []
        for segment in segments:
            if len(segment) > max_length:
                # Split long segments at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', segment)
                current_subsegment = ""
                
                for sentence in sentences:
                    if len(current_subsegment) + len(sentence) + 1 <= max_length:
                        if current_subsegment:
                            current_subsegment += " " + sentence
                        else:
                            current_subsegment = sentence
                    else:
                        if current_subsegment and len(current_subsegment) >= min_length:
                            final_segments.append(current_subsegment.strip())
                        current_subsegment = sentence
                
                if current_subsegment and len(current_subsegment) >= min_length:
                    final_segments.append(current_subsegment.strip())
            else:
                final_segments.append(segment)
        
        return final_segments
    
    
    def save_statistics(self):
        """Save processing statistics"""
        stats_file = config.london_historical_data / "unified_processing_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to: {stats_file}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 80)
        print("HISTORICAL DATA COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Sources processed: {self.stats['sources_processed']}")
        print(f"Files downloaded: {self.stats['files_downloaded']}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files cleaned: {self.stats['files_cleaned']}")
        print(f"Files sanitized: {self.stats['files_sanitized']}")
        print(f"Files skipped: {self.stats['files_skipped']}")
        print(f"Files failed: {self.stats['files_failed']}")
        print(f"Total characters before: {self.stats['total_chars_before']:,}")
        print(f"Total characters after: {self.stats['total_chars_after']:,}")
        print(f"Characters removed: {self.stats['chars_removed']:,}")
        print(f"Gutenberg headers removed: {self.stats['gutenberg_headers_removed']}")
        print(f"OCR artifacts fixed: {self.stats['ocr_artifacts_fixed']}")
        print(f"HTML markup removed: {self.stats['html_markup_removed']}")
        print(f"Duplicates found: {self.stats['duplicates_found']}")
        print(f"Non-English skipped: {self.stats['non_english_skipped']}")
        print(f"Poor quality skipped: {self.stats['poor_quality_skipped']}")
        print(f"Gutenberg processed: {self.stats['gutenberg_processed']}")
        print(f"Gutenberg accepted: {self.stats['gutenberg_accepted']}")
        print("=" * 80)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Data Manager for London Historical LLM")
    parser.add_argument("--max_sources", type=int, default=None,
                       help="Maximum number of sources to process")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory for processed data")
    parser.add_argument("--sanitize_only", action="store_true",
                       help="Only sanitize existing filenames, don't download or process")
    parser.add_argument("--clean_existing", action="store_true",
                       help="Clean repetitive patterns from existing processed data")
    # Data source configuration is handled in config.py
    # Command line overrides removed for cleaner code management
    
    args = parser.parse_args()
    
    # Configuration is already set in config.py - no need to override
    # Command line arguments are kept for backward compatibility but not used
    
    # Create data collector
    collector = HistoricalDataCollector()
    
    if args.sanitize_only:
        # Only sanitize filenames
        print("🔧 Running filename sanitization only...")
        sanitized_count = collector.sanitize_existing_filenames()
        print(f"\n✅ Sanitized {sanitized_count} filenames")
        print("📁 All filenames are now safe for processing")
        return True
    elif args.clean_existing:
        # Clean existing processed data
        print("🧹 Cleaning repetitive patterns from existing processed data...")
        success = collector.clean_existing_processed_data()
        if success:
            print("\n✅ Data cleaning completed successfully!")
            print("📚 Processed data is now clean and ready for training")
        else:
            print("\n❌ Data cleaning failed")
        return success
    else:
        # Run full processing
        success = collector.download_and_process_sources(args.max_sources)
        
        if success:
            collector.print_summary()
            print("\n✅ Historical data collection completed successfully!")
            print("📚 Comprehensive corpus created and ready for training")
        else:
            print("\n❌ Historical data collection failed")
            print("📋 Check historical_data_collector.log for error details")
        
        return success

if __name__ == "__main__":
    main()
