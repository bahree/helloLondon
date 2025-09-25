# London Historical Data Collection Guide

This guide explains how to collect, process, and manage historical data for the London Historical LLM project.

## **Overview**

> **🚀 Published Model**: The data collected by this system has been used to train the [London Historical SLM](https://huggingface.co/bahree/london-historical-slm) - a 117M parameter model available on Hugging Face.

The data collection system provides comprehensive historical text data spanning **1500-1850** with:

- **218+ historical sources** covering 350 years of London history
- **500M+ characters** of authentic historical English text
- **Modular design**: Configuration-based, easy to expand
- **Professional logging**: Comprehensive statistics and monitoring
- **Quality assurance**: Automated cleaning and validation
- **Archive.org integration**: Direct API access to historical collections

## **Dataset Statistics**

| **Metric** | **Value** |
|------------|-----------|
| **Total Sources** | 218+ historical texts |
| **Time Period** | 1500-1850 (350 years) |
| **Estimated Characters** | 500M+ characters |
| **Estimated Tokens** | ~125M tokens |
| **Languages** | Historical English (1500-1850) |
| **Geographic Focus** | London and England |
| **Text Types** | Literature, legal, religious, scientific, commercial, personal |

## Quick Start

### 1. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu)
sudo apt-get install poppler-utils
```

### 2. Check Access Requirements
> **📋 Important**: Some data sources require registration. See [Access Requirements](02_data_collection/ACCESS_REQUIREMENTS.md) for detailed information about free vs. paid sources and registration instructions.

### 3. Run Data Collection
```bash
cd 02_data_collection
python historical_data_collector.py
```

### 4. Add New Data Sources
```bash
python add_data_source.py
```

### 5. Check Failed Downloads
```bash
# Quick analysis of failed downloads
python3 check_failed_downloads.py

# Generate detailed reports
python3 generate_report.py --format csv
```

## System Architecture

### Files Structure
```
02_data_collection/
├── data_sources.json           # Configuration file with all data sources
├── historical_data_collector.py # Main data collection and processing script
├── add_data_source.py          # Add new sources interactively
├── generate_report.py          # Generate detailed reports
└── check_failed_downloads.py   # Quick failed downloads analysis
```

### Configuration File
The `data_sources.json` file contains:
- **Historical Sources**: 12+ configured sources with metadata
- **Gutenberg Sources**: 8+ Project Gutenberg texts
- **Source Properties**: URLs, time periods, formats, priorities

### Logging and Reports

## 📊 Failed Downloads Analysis

### Quick Analysis
```bash
# Check failed downloads with summary
python3 check_failed_downloads.py
```

**Output includes:**
- Total sources attempted/successful/failed
- List of all failed downloads with details
- Error type summary and counts
- Source breakdown

### Detailed Reports
```bash
# Generate CSV report of failed downloads
python3 generate_report.py --format csv

# Generate all report formats
python3 generate_report.py --format all

# Generate specific format
python3 generate_report.py --format json
```

### Manual Analysis
```bash
# Check statistics file directly
cat data/london_historical/download_statistics.json | jq '.failed_downloads | length'

# List failed downloads with URLs
python3 -c "
import json
with open('data/london_historical/download_statistics.json') as f:
    stats = json.load(f)
for failure in stats.get('failed_downloads', []):
    print(f'{failure.get(\"source\")} | {failure.get(\"filename\")} | {failure.get(\"url\")} | {failure.get(\"error\")}')
"
```

### Common Error Types
- **404 Not Found**: URL doesn't exist (check URL)
- **Connection timeout**: Network issues (retry later)
- **403 Forbidden**: Access denied (check credentials)
- **SSL errors**: Certificate issues (update certificates)

## 🔧 Troubleshooting Failed Downloads

### Quick Diagnosis
```bash
# Check overall statistics
python3 check_failed_downloads.py

# Look for specific error patterns
python3 check_failed_downloads.py | grep "404"
python3 check_failed_downloads.py | grep "timeout"
```

### Fix Common Issues

#### 1. 404 Not Found Errors
```bash
# Check if URLs are still valid
curl -I "https://example.com/file.txt"

# Update URLs in data_sources.json
# Re-run download for specific sources
```

#### 2. Connection Timeouts
```bash
# Check network connectivity
ping google.com

# Increase timeout in data_downloader.py
# Retry failed downloads
```

#### 3. SSL Certificate Errors
```bash
# Update certificates
sudo apt-get update && sudo apt-get install ca-certificates

# Or disable SSL verification (not recommended)
```

### Retry Failed Downloads
```bash
# Re-run data collection (will skip successful downloads)
python3 historical_data_collector.py

# Or run specific sources only
python3 historical_data_collector.py --sources "Project Gutenberg"
```

### Manual URL Fixes
1. **Check URLs manually** in browser
2. **Update data_sources.json** with correct URLs
3. **Re-run download** for specific sources
4. **Monitor progress** with check_failed_downloads.py

### Logging and Reports
- **Log File**: `historical_data_collector.log`
- **Statistics**: `data/london_historical/download_statistics.json`
- **Reports**: Text, JSON, and CSV formats available

## 🚨 **Comprehensive Failed Downloads Handling**

### **Real-time Error Monitoring**
The `historical_data_collector.py` provides comprehensive error tracking:

```bash
# Monitor errors in real-time
tail -f historical_data_collector.log | grep -E "ERROR|FAILED|❌"

# Check error summary
grep -c "❌\|ERROR\|FAILED" historical_data_collector.log
```

### **Error Types and Solutions**

#### **1. Download Failures**
| Error | Cause | Solution |
|-------|-------|----------|
| `404 Not Found` | URL doesn't exist | Check URL manually, update `data_sources.json` |
| `Connection timeout` | Network issues | Retry later, check internet connection |
| `403 Forbidden` | Access denied | Check if site requires authentication |
| `SSL errors` | Certificate issues | Update certificates: `sudo apt-get install ca-certificates` |

#### **2. Processing Failures**
| Error | Cause | Solution |
|-------|-------|----------|
| `PDF extraction failed` | Corrupted PDF | Skip file, try alternative source |
| `HTML extraction failed` | Malformed HTML | Check HTML structure, update parser |
| `Encoding issues` | Character encoding | System handles automatically |
| `Unsupported format` | Unknown file type | Add format support or skip |

#### **3. Content Filtering**
| Filter | Reason | Action |
|--------|--------|--------|
| `Non-English text` | Language detection | Skipped automatically |
| `Duplicate content` | Content-based dedup | Skipped automatically |
| `Too short` | Quality filter | Skipped automatically |

### **Advanced Error Recovery**

#### **Check Specific Error Types**
```bash
# Network-related errors
grep -E "timeout|connection|network" historical_data_collector.log

# File processing errors
grep -E "PDF|HTML|extraction" historical_data_collector.log

# URL-related errors
grep -E "404|403|SSL" historical_data_collector.log
```

#### **Generate Error Reports**
```bash
# Create detailed error report
python3 -c "
import json
with open('data/london_historical/download_statistics.json') as f:
    stats = json.load(f)

print('=== FAILED DOWNLOADS REPORT ===')
print(f'Total sources: {stats.get(\"total_sources\", 0)}')
print(f'Successful: {stats.get(\"successful_downloads\", 0)}')
print(f'Failed: {stats.get(\"failed_downloads\", 0)}')
print(f'Success rate: {stats.get(\"success_rate\", 0):.1f}%')
print()

print('=== FAILURE BREAKDOWN ===')
failures = stats.get('failed_downloads', [])
error_types = {}
for failure in failures:
    error = failure.get('error', 'Unknown')
    error_types[error] = error_types.get(error, 0) + 1

for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
    print(f'{error}: {count} failures')

print()
print('=== DETAILED FAILURES ===')
for failure in failures[:10]:  # Show first 10
    print(f'Source: {failure.get(\"source\", \"Unknown\")}')
    print(f'File: {failure.get(\"filename\", \"Unknown\")}')
    print(f'URL: {failure.get(\"url\", \"Unknown\")}')
    print(f'Error: {failure.get(\"error\", \"Unknown\")}')
    print('---')
"

# Save to file
python3 -c "
import json
with open('data/london_historical/download_statistics.json') as f:
    stats = json.load(f)
with open('failed_downloads_report.txt', 'w') as f:
    f.write('FAILED DOWNLOADS REPORT\n')
    f.write('=' * 50 + '\n\n')
    for failure in stats.get('failed_downloads', []):
        f.write(f'Source: {failure.get(\"source\", \"Unknown\")}\n')
        f.write(f'File: {failure.get(\"filename\", \"Unknown\")}\n')
        f.write(f'URL: {failure.get(\"url\", \"Unknown\")}\n')
        f.write(f'Error: {failure.get(\"error\", \"Unknown\")}\n')
        f.write('-' * 30 + '\n')
print('Report saved to failed_downloads_report.txt')
"
```

#### **Targeted Retry Strategies**

##### **Retry All Failed Downloads**
```bash
# Re-run with focus on failed sources
python3 historical_data_collector.py --retry-failed
```

##### **Retry Specific Source Types**
```bash
# Retry only Project Gutenberg sources
python3 historical_data_collector.py --sources "Project Gutenberg"

# Retry only PDF sources
python3 historical_data_collector.py --sources "PDF"

# Retry only HTML sources
python3 historical_data_collector.py --sources "HTML"
```

##### **Retry with Increased Timeouts**
```bash
# For timeout issues, modify the script temporarily
# Or run with system-level timeout increase
timeout 3600 python3 historical_data_collector.py
```

### **Manual URL Fixes**

#### **1. Identify Problematic URLs**
```bash
# Extract all failed URLs
python3 -c "
import json
with open('data/london_historical/download_statistics.json') as f:
    stats = json.load(f)
failed_urls = [f.get('url') for f in stats.get('failed_downloads', []) if f.get('url')]
for url in failed_urls:
    print(url)
" > failed_urls.txt
```

#### **2. Test URLs Manually**
```bash
# Test each URL
while read url; do
    echo "Testing: $url"
    curl -I "$url" 2>/dev/null | head -1
    echo "---"
done < failed_urls.txt
```

#### **3. Update data_sources.json**
```bash
# Edit the configuration file
nano data_sources.json

# Or use sed for bulk replacements
sed -i 's/old_url/new_url/g' data_sources.json
```

### **Quality Control and Validation**

#### **Check Processing Success Rate**
```bash
# Overall success rate
python3 -c "
import json
with open('data/london_historical/download_statistics.json') as f:
    stats = json.load(f)
total = stats.get('total_sources', 0)
successful = stats.get('successful_downloads', 0)
failed = stats.get('failed_downloads', 0)
print(f'Success rate: {successful/total*100:.1f}%' if total > 0 else 'No data')
print(f'Failed rate: {failed/total*100:.1f}%' if total > 0 else 'No data')
"
```

#### **Validate Downloaded Content**
```bash
# Check file sizes and content
cd data/london_historical/
echo "=== FILE SIZE ANALYSIS ==="
ls -lh *.txt *.pdf *.html 2>/dev/null | awk '{print $5, $9}' | sort -hr

echo "=== EMPTY FILES ==="
find . -name "*.txt" -size 0 -o -name "*.pdf" -size 0 -o -name "*.html" -size 0

echo "=== CONTENT VALIDATION ==="
for file in *.txt; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        chars=$(wc -c < "$file")
        echo "$file: $lines lines, $chars characters"
    fi
done
```

### **Emergency Recovery**

#### **If Data Collection Completely Fails**
```bash
# 1. Check system resources
df -h  # Disk space
free -h  # Memory
nvidia-smi  # GPU (if using)

# 2. Check network connectivity
ping google.com
curl -I https://www.gutenberg.org

# 3. Check Python environment
python3 --version
pip list | grep -E "requests|beautifulsoup4|PyPDF2"

# 4. Start with minimal sources
python3 historical_data_collector.py --max_sources 5
```

#### **Partial Recovery**
```bash
# Continue from where it left off
python3 historical_data_collector.py

# The system automatically skips already downloaded files
```

### **Best Practices for Error Prevention**

1. **Run during off-peak hours** to avoid network congestion
2. **Use stable internet connection** (avoid mobile hotspots)
3. **Monitor disk space** (ensure 5GB+ free space)
4. **Check system resources** before starting
5. **Run in screen/tmux** for long-running processes
6. **Keep logs** for troubleshooting

### **Success Metrics**

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Overall Success Rate** | >90% | 75-90% | <75% |
| **Text Files** | >95% | 85-95% | <85% |
| **PDF Files** | >80% | 60-80% | <60% |
| **HTML Files** | >85% | 70-85% | <70% |

## Adding New Data Sources

### Interactive Method
```bash
python add_data_source.py
```
Follow the prompts to add new sources.

### Manual Method
Edit `data_sources.json` directly:
```json
{
  "historical_sources": {
    "new_source": {
      "name": "New Source Name",
      "description": "Source description",
      "time_period": [1500, 1850],
      "format": "TXT",
      "url": "https://example.com",
      "license": "Public Domain",
      "type": "historical_text",
      "priority": "medium",
      "scraping_enabled": true,
      "search_terms": ["London", "historical"]
    }
  }
}
```

## Generated Reports

### Text Report
```bash
python generate_report.py --format text
```
Creates: `data_collection_report.txt`

### JSON Report
```bash
python generate_report.py --format json
```
Creates: `data_collection_report.json`

### CSV Report (Failed Downloads)
```bash
python generate_report.py --format csv
```
Creates: `failed_downloads.csv`

### All Reports
```bash
python generate_report.py --format all
```

## Check Results
```bash
ls -la data/london_historical/
```

### 3. View Statistics
```bash
cat data/london_historical/download_statistics.json
```

## Data Sources

### Project Gutenberg Sources
- **Time Period**: 1500-1850
- **Format**: Plain text (.txt)
- **Content**: Novels, historical accounts, letters
- **Examples**: Defoe's "A Journal of the Plague Year", historical surveys

### Historical Archives
- **London Lives**: 240,000 manuscript pages (1690-1800)
- **Old Bailey**: 197,000+ trial accounts (1674-1850)
- **TNA Records**: Government correspondence and records
- **British History Online**: Historical surveys and documents

## Data Processing

### Text Cleaning
The system automatically processes files through a comprehensive cleaning pipeline:

**File Type Processing:**
- **Text files** (.txt, .txt.utf-8): Project Gutenberg header removal, metadata cleanup
- **PDF files**: OCR text extraction, page number removal, OCR artifact correction
- **HTML files**: BeautifulSoup parsing, unwanted element removal, wiki metadata cleanup
- **XML files**: Specialized processing for Old Bailey and London Lives with historical language preservation

**Quality Validation:**
- Duplicate content detection (hash-based)
- Language detection and filtering (English only)
- OCR artifact detection and correction
- Advertisement content filtering
- Meaningful word ratio analysis (≥50% for general content)

**Historical Text Preservation:**
- Old Bailey XML: Trial accounts, front matter, historical language patterns
- London Lives XML: Semantic markup preservation, person names, places, occupations
- Project Gutenberg: Relaxed quality criteria for historical content

**Complete Process**: See [Cleaning Process Flow](CLEANING_PROCESS_FLOW.md) for detailed implementation.

### File Organization
```
data/london_historical/
├── london_historical_corpus.txt    # Main training corpus
├── gutenberg_*.txt                 # Individual Gutenberg texts
├── london_lives_processed.txt      # London Lives data
├── old_bailey_processed.txt        # Old Bailey data
├── failed_downloads.json           # Failed downloads log
├── manual_retry_script.py          # Retry script
└── download_statistics.json        # Download statistics
```

## Failed Downloads

### Automatic Retry
The system automatically retries failed downloads with exponential backoff.

### Manual Retry
For persistent failures:

1. **Check failed downloads**:
   ```bash
   python manual_retry_helper.py
   ```

2. **Generate retry script**:
   ```bash
   python manual_retry_helper.py data/london_historical generate
   ```

3. **Run retry script**:
   ```bash
   python manual_retry_helper.py data/london_historical run
   ```

### Retry Commands
The system generates multiple retry options:
- Python script: `manual_retry_script.py`
- Curl commands: `manual_retry_curl.sh`
- Wget commands: `manual_retry_wget.sh`

## Configuration

### Time Period Filtering
```python
# Default: 1500-1850
downloader = LondonHistoricalDataDownloader(time_period=(1500, 1850))

# Custom period
downloader = LondonHistoricalDataDownloader(time_period=(1600, 1800))
```

### Source Selection
```python
# Enable/disable specific sources
downloader.historical_sources['defoe_plague']['enabled'] = True
downloader.gutenberg_sources = [source for source in downloader.gutenberg_sources if source['year'] >= 1600]
```

## Monitoring Progress

### Log Files
- `data_collection.log`: Detailed download logs
- `download_statistics.json`: Download statistics
- `failed_downloads.json`: Failed download details

### Progress Tracking
The system shows:
- Real-time download progress
- Success/failure counts
- File sizes and download speeds
- Estimated completion time

## Troubleshooting

### Common Issues

1. **Network Timeouts**
   - Check internet connection
   - Increase timeout values
   - Use manual retry for specific files

2. **Permission Errors**
   - Ensure write permissions to data directory
   - Run as administrator if needed

3. **Disk Space**
   - Monitor available space
   - Clean up old downloads if needed

4. **Rate Limiting**
   - The system includes delays between requests
   - Use manual retry for rate-limited sources

### Getting Help

- Check logs: `tail -f data_collection.log`
- Run diagnostics: `python 06_testing/test_system.py`
- Review failed downloads: `cat data/london_historical/failed_downloads.json`

## Data Quality

### Text Preprocessing
- Removes metadata headers
- Normalizes historical spelling
- Preserves historical language patterns
- Filters by relevance and quality

### Validation
- Checks file integrity
- Validates text content
- Ensures proper encoding
- Verifies time period relevance

## Expected Results

### Data Volume
- **Total Size**: 500MB - 2GB
- **Text Sources**: 50-200 files
- **Character Count**: 10-50 million characters
- **Time Coverage**: 350 years (1500-1850)

### Quality Metrics
- **Success Rate**: 70-90% (depending on source availability)
- **Text Quality**: High (preprocessed and validated)
- **Historical Accuracy**: Verified against source metadata
- **Language Diversity**: Covers multiple historical periods

## Next Steps

After data collection:

1. **Verify Data**: Check corpus file and statistics
2. **Train Tokenizer**: `cd 03_tokenizer && python train_tokenizer.py`
3. **Prepare Dataset**: The tokenizer will process the collected data
4. **Start Training**: `cd 04_training && python train_model.py`

## Advanced Usage

### Custom Sources
Add new data sources by modifying the `historical_sources` dictionary in the downloader script.

### Batch Processing
For large-scale collection, consider running multiple instances with different source subsets.

### Remote Execution
The system is optimized for remote server execution with robust error handling and retry mechanisms.
