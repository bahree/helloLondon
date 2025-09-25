#!/usr/bin/env python3
"""
Filename Sanitizer for Historical Documents
==========================================
Renames downloaded files to safer, shorter names without special characters
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import global configuration
from config import config

class FilenameSanitizer:
    """Sanitizes filenames to be safe for processing"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or config.london_historical_data
        self.renamed_files = {}
        
    def sanitize_filename(self, original_name: str, max_length: int = 100) -> str:
        """Convert a filename to a safe, shorter version"""
        
        # Remove file extension
        name, ext = os.path.splitext(original_name)
        
        # Create a safe base name
        safe_name = self._create_safe_name(name, max_length - len(ext))
        
        # Ensure uniqueness
        final_name = self._ensure_unique_filename(safe_name + ext)
        
        return final_name
    
    def _create_safe_name(self, name: str, max_length: int) -> str:
        """Create a safe filename from the original name"""
        
        # Remove or replace problematic characters
        safe_name = re.sub(r'[^\w\s\-_]', '_', name)  # Replace special chars with underscore
        safe_name = re.sub(r'[,\;\.]+', '_', safe_name)  # Replace commas, semicolons, periods
        safe_name = re.sub(r'_{2,}', '_', safe_name)  # Replace multiple underscores with single
        safe_name = safe_name.strip('_')  # Remove leading/trailing underscores
        
        # Truncate if too long
        if len(safe_name) > max_length:
            # Try to truncate at word boundary
            truncated = safe_name[:max_length]
            last_underscore = truncated.rfind('_')
            if last_underscore > max_length * 0.7:  # If we can keep 70% of the name
                safe_name = truncated[:last_underscore]
            else:
                safe_name = truncated
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = "unnamed_document"
        
        return safe_name
    
    def _ensure_unique_filename(self, filename: str) -> str:
        """Ensure the filename is unique by adding a number if needed"""
        
        base_name, ext = os.path.splitext(filename)
        counter = 1
        final_name = filename
        
        while final_name in self.renamed_files.values():
            final_name = f"{base_name}_{counter}{ext}"
            counter += 1
        
        return final_name
    
    def sanitize_directory(self, directory: Path = None) -> Dict[str, str]:
        """Sanitize all filenames in a directory"""
        
        if directory is None:
            directory = self.data_dir
        
        print(f"🔧 Sanitizing filenames in: {directory}")
        
        # Find all text files
        text_files = list(directory.glob("*.txt"))
        
        if not text_files:
            print("❌ No text files found to sanitize")
            return {}
        
        print(f"📁 Found {len(text_files)} text files to sanitize")
        
        renamed_count = 0
        
        for file_path in text_files:
            original_name = file_path.name
            sanitized_name = self.sanitize_filename(original_name)
            
            if original_name != sanitized_name:
                # Rename the file
                new_path = file_path.parent / sanitized_name
                
                try:
                    file_path.rename(new_path)
                    self.renamed_files[original_name] = sanitized_name
                    renamed_count += 1
                    print(f"✅ {original_name} → {sanitized_name}")
                except Exception as e:
                    print(f"❌ Failed to rename {original_name}: {e}")
            else:
                print(f"⏭️  {original_name} (no change needed)")
        
        print(f"\n📊 Sanitization complete: {renamed_count} files renamed")
        return self.renamed_files
    
    def create_rename_mapping_file(self, output_file: str = "filename_mapping.json"):
        """Save the mapping of old to new filenames"""
        
        if not self.renamed_files:
            print("No files were renamed")
            return
        
        mapping_file = self.data_dir / output_file
        
        import json
        with open(mapping_file, 'w') as f:
            json.dump(self.renamed_files, f, indent=2)
        
        print(f"💾 Filename mapping saved to: {mapping_file}")
    
    def show_filename_statistics(self, directory: Path = None):
        """Show statistics about filenames in the directory"""
        
        if directory is None:
            directory = self.data_dir
        
        text_files = list(directory.glob("*.txt"))
        
        if not text_files:
            print("❌ No text files found")
            return
        
        print(f"\n📊 Filename Statistics for {directory}")
        print("=" * 50)
        
        # Length statistics
        lengths = [len(f.name) for f in text_files]
        print(f"Total files: {len(text_files)}")
        print(f"Average filename length: {sum(lengths) / len(lengths):.1f} characters")
        print(f"Shortest filename: {min(lengths)} characters")
        print(f"Longest filename: {max(lengths)} characters")
        
        # Problematic characters
        problematic_chars = [',', ';', ':', '(', ')', '[', ']', '{', '}', '&', '%', '#', '@', '!', '?']
        files_with_problems = []
        
        for file_path in text_files:
            for char in problematic_chars:
                if char in file_path.name:
                    files_with_problems.append(file_path.name)
                    break
        
        print(f"Files with special characters: {len(files_with_problems)}")
        
        if files_with_problems:
            print("\nFiles with problematic characters:")
            for filename in files_with_problems[:10]:  # Show first 10
                print(f"  - {filename}")
            if len(files_with_problems) > 10:
                print(f"  ... and {len(files_with_problems) - 10} more")

def main():
    """Main function for filename sanitization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanitize filenames in data directory")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory to sanitize (defaults to config)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be renamed without actually renaming")
    parser.add_argument("--stats_only", action="store_true",
                       help="Only show filename statistics")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else config.london_historical_data
    
    sanitizer = FilenameSanitizer(data_dir)
    
    if args.stats_only:
        sanitizer.show_filename_statistics()
        return
    
    if args.dry_run:
        print("🔍 DRY RUN - No files will be renamed")
        sanitizer.show_filename_statistics()
        return
    
    # Show current statistics
    sanitizer.show_filename_statistics()
    
    # Ask for confirmation
    response = input("\n🤔 Proceed with filename sanitization? (y/N): ")
    if response.lower() != 'y':
        print("❌ Sanitization cancelled")
        return
    
    # Perform sanitization
    renamed_files = sanitizer.sanitize_directory()
    
    if renamed_files:
        sanitizer.create_rename_mapping_file()
        print(f"\n✅ Sanitization complete! {len(renamed_files)} files renamed")
    else:
        print("\n✅ No files needed sanitization")

if __name__ == "__main__":
    main()
