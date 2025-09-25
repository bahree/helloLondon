#!/usr/bin/env python3
"""
Optimized Archive.org Data Collector for London Historical LLM
Uses internetarchive package for better performance and reliability
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Internet Archive library
try:
    import internetarchive as ia
    IA_AVAILABLE = True
except ImportError:
    IA_AVAILABLE = False
    print("Warning: internetarchive package not available. Install with: pip install internetarchive")
    import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config

class OptimizedArchiveOrgCollector:
    """Optimized Archive.org data collector using internetarchive package"""
    
    def __init__(self):
        self.setup_logging()
        
        if not IA_AVAILABLE:
            self.logger.error("internetarchive package not available. Please install it first.")
            raise ImportError("internetarchive package required")
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        # Output directories
        self.download_dir = config.london_historical_data / "downloads" / "archive_org_optimized"
        self.metadata_dir = config.london_historical_data / "metadata" / "archive_org_optimized"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'searches_performed': 0,
            'items_found': 0,
            'items_downloaded': 0,
            'items_skipped': 0,
            'items_failed': 0,
            'total_size_downloaded': 0,
            'collections_processed': []
        }
        
        self.logger.info("Optimized Archive.org collector initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('archive_org_collector.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def search_archive(self, query: str, filters: Dict[str, Any] = None, 
                      rows: int = 50) -> List[Dict[str, Any]]:
        """
        Search Archive.org using internetarchive package
        
        Args:
            query: Search query string
            filters: Additional filters (date range, format, etc.)
            rows: Number of results per page
        
        Returns:
            List of search results with metadata
        """
        self.rate_limit()
        
        try:
            # Build search query with filters
            search_query = query
            if filters:
                for key, value in filters.items():
                    if key == 'date_range':
                        start_date, end_date = value
                        search_query += f' AND date:[{start_date} TO {end_date}]'
                    elif key == 'format':
                        search_query += f' AND format:{value}'
                    elif key == 'language':
                        search_query += f' AND language:{value}'
                    elif key == 'collection':
                        search_query += f' AND collection:{value}'
            
            self.logger.info(f"Searching Archive.org: {search_query}")
            
            # Use internetarchive search
            search = ia.search_items(search_query)
            items = []
            count = 0
            for item in search:
                items.append(item)  # item is a dictionary
                count += 1
                if count >= rows:  # Limit results
                    break
            
            self.stats['searches_performed'] += 1
            self.stats['items_found'] += len(items)
            
            self.logger.info(f"Found {len(items)} items")
            return items
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_item_metadata(self, identifier: str) -> Dict[str, Any]:
        """
        Get detailed metadata for an Archive.org item
        
        Args:
            identifier: Archive.org item identifier
            
        Returns:
            Dictionary containing item metadata
        """
        self.rate_limit()
        
        try:
            item = ia.get_item(identifier)
            metadata = item.metadata
            
            return {
                'identifier': identifier,
                'title': metadata.get('title', 'Unknown'),
                'creator': metadata.get('creator', 'Unknown'),
                'date': metadata.get('date', 'Unknown'),
                'description': metadata.get('description', ''),
                'subject': metadata.get('subject', []),
                'language': metadata.get('language', 'en'),
                'collection': metadata.get('collection', []),
                'files': list(item.files),
                'file_count': len(item.files)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {identifier}: {e}")
            return {}
    
    def get_item_metadata_from_search(self, search_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata from a search result item (dictionary)
        
        Args:
            search_item: Dictionary from search_items() result
            
        Returns:
            Dictionary containing item metadata
        """
        return {
            'identifier': search_item.get('identifier', 'Unknown'),
            'title': search_item.get('title', 'Unknown'),
            'creator': search_item.get('creator', 'Unknown'),
            'date': search_item.get('date', 'Unknown'),
            'description': search_item.get('description', ''),
            'subject': search_item.get('subject', []),
            'language': search_item.get('language', 'en'),
            'collection': search_item.get('collection', []),
            'files': [],  # Will be populated when we get full item
            'file_count': 0
        }
    
    def find_text_files(self, item_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find downloadable text files in an Archive.org item
        
        Args:
            item_metadata: Item metadata dictionary
            
        Returns:
            List of text file information dictionaries
        """
        text_files = []
        files = item_metadata.get('files', [])
        
        for file_info in files:
            name = file_info.get('name', '')
            format_type = file_info.get('format', '')
            size = file_info.get('size', 0)
            
            # Check for various text file types
            if (name.endswith('.txt') and 
                format_type in ['Plain Text', 'Text', 'DjVuTXT', ''] and
                int(size) > 1000 and  # At least 1KB
                not name.endswith('_meta.txt')):  # Skip metadata files
                text_files.append(file_info)
            elif name.endswith('.djvu.txt'):
                text_files.append(file_info)
        
        # Sort by preference (plain text first, then by size)
        text_files.sort(key=lambda x: (
            x.get('format', '') != 'Plain Text',  # Plain text first
            -int(x.get('size', 0))  # Larger files first
        ))
        
        return text_files
    
    def download_text_file(self, identifier: str, filename: str, 
                          output_path: Path) -> bool:
        """
        Download a text file from an Archive.org item
        
        Args:
            identifier: Archive.org item identifier
            filename: Name of the file to download
            output_path: Local path to save the file
            
        Returns:
            True if download successful
        """
        self.rate_limit()
        
        try:
            item = ia.get_item(identifier)
            file_obj = item.get_file(filename)
            
            if file_obj:
                # Create a safe filename by sanitizing it
                safe_filename = self.sanitize_filename(filename)
                safe_output_path = output_path.parent / safe_filename
                
                # Download the file to a temporary location first (within current working directory)
                temp_path = Path.cwd() / f"temp_{safe_filename}"
                
                try:
                    # Download to temp location
                    file_obj.download(str(temp_path))
                    
                    # Move to final location
                    temp_path.rename(safe_output_path)
                    
                    # Verify download
                    if safe_output_path.exists() and safe_output_path.stat().st_size > 0:
                        self.stats['total_size_downloaded'] += safe_output_path.stat().st_size
                        return True
                    else:
                        self.logger.warning(f"Downloaded file is empty or missing: {safe_output_path}")
                        return False
                        
                except Exception as e:
                    # Clean up temp file if it exists
                    if temp_path.exists():
                        temp_path.unlink()
                    raise e
            else:
                self.logger.warning(f"File not found: {filename} in {identifier}")
                return False
                
        except Exception as e:
            self.logger.error(f"Download failed for {filename}: {e}")
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem"""
        import re
        
        # Remove or replace problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_name = re.sub(r'[^\w\-_\.]', '_', safe_name)
        
        # Limit length
        if len(safe_name) > 200:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:200-len(ext)] + ext
        
        return safe_name
    
    def collect_london_sources(self, max_items: int = 100) -> bool:
        """
        Collect London-related historical texts from Archive.org
        
        Args:
            max_items: Maximum number of items to process
            
        Returns:
            True if collection successful
        """
        self.logger.info("🔍 Starting optimized London historical data collection...")
        
        # Define search queries for different collections
        search_queries = [
            {
                'query': 'collection:"philosophicaltransactions" AND mediatype:texts AND date:[1800 TO 1850]',
                'name': 'Philosophical Transactions',
                'max_items': min(max_items, 1000)
            },
            {
                'query': 'subject:"London" AND mediatype:texts AND date:[1800 TO 1850]',
                'name': 'London Historical Texts',
                'max_items': min(max_items, 500)
            },
            {
                'query': 'creator:"Royal Society" AND mediatype:texts AND date:[1800 TO 1850]',
                'name': 'Royal Society Publications',
                'max_items': min(max_items, 300)
            }
        ]
        
        total_downloaded = 0
        
        for search_config in search_queries:
            query = search_config['query']
            collection_name = search_config['name']
            max_collection_items = search_config['max_items']
            
            self.logger.info(f"📚 Processing collection: {collection_name}")
            
            # Search for items
            items = self.search_archive(query, rows=max_collection_items)
            
            if not items:
                self.logger.warning(f"No items found for {collection_name}")
                continue
            
            # Process items
            collection_downloaded = 0
            for i, item in enumerate(items[:max_collection_items]):
                identifier = item.get('identifier', '')
                if not identifier:
                    continue
                
                self.logger.info(f"📄 Processing item {i+1}/{min(len(items), max_collection_items)}: {identifier}")
                
                # Get detailed metadata
                metadata = self.get_item_metadata(identifier)
                if not metadata:
                    self.stats['items_failed'] += 1
                    continue
                
                # Find text files
                text_files = self.find_text_files(metadata)
                
                if not text_files:
                    self.logger.info(f"No text files found in {identifier}")
                    self.stats['items_skipped'] += 1
                    continue
                
                # Download text files
                item_downloaded = 0
                for file_info in text_files:
                    filename = file_info.get('name', '')
                    if not filename:
                        continue
                    
                    # Create output filename with safe sanitization
                    safe_identifier = self.sanitize_filename(identifier)
                    safe_filename = self.sanitize_filename(filename)
                    output_filename = f"{safe_identifier}_{safe_filename}"
                    output_path = self.download_dir / output_filename
                    
                    # Skip if already downloaded
                    if output_path.exists():
                        self.logger.info(f"File already exists: {output_filename}")
                        item_downloaded += 1
                        continue
                    
                    # Download file
                    self.logger.info(f"⬇️  Downloading: {filename}")
                    if self.download_text_file(identifier, filename, output_path):
                        item_downloaded += 1
                        self.stats['items_downloaded'] += 1
                        self.logger.info(f"✅ Downloaded: {filename}")
                    else:
                        self.stats['items_failed'] += 1
                
                if item_downloaded > 0:
                    collection_downloaded += 1
                    total_downloaded += 1
                
                # Save metadata
                metadata_path = self.metadata_dir / f"{identifier}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.stats['collections_processed'].append({
                'name': collection_name,
                'items_found': len(items),
                'items_downloaded': collection_downloaded
            })
            
            self.logger.info(f"✅ {collection_name}: Downloaded {collection_downloaded} items")
        
        # Save final statistics
        stats_path = self.metadata_dir / "collection_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"🎉 Collection completed! Downloaded {total_downloaded} items")
        self.logger.info(f"📊 Total size: {self.stats['total_size_downloaded']:,} bytes")
        
        return total_downloaded > 0

def main():
    """Main function for testing"""
    collector = OptimizedArchiveOrgCollector()
    
    # Test with small number of items
    success = collector.collect_london_sources(max_items=5)
    
    if success:
        print("✅ Test completed successfully!")
        print(f"📁 Files saved to: {collector.download_dir}")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()
