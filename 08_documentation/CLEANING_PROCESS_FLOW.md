# 🧹 Historical Text Cleaning Process Flow

This document shows the **actual** text cleaning process implemented in `historical_data_collector.py`, not just theoretical cleaning steps.

> **📊 Visual Flow**: See [Cleaning Flow Diagram](cleaning_flow_diagram.md) for a Mermaid diagram of the complete process.

## 📊 **Complete Cleaning Pipeline**

### **Phase 1: File Discovery & Initial Filtering**

```
📁 Raw Files
    ↓
🔍 File Type Detection
    ├── .txt, .txt.utf-8, _txt.utf-8 → Text Processing
    ├── .pdf → PDF Processing  
    ├── .html, .htm → HTML Processing
    ├── .xml → XML Processing (Old Bailey, London Lives)
    └── No Extension → Content Detection
        ├── HTML-like content → HTML Processing
        ├── Text-like content → Text Processing
        └── Binary/Unknown → REJECTED
    ↓
🚫 Filename Language Check
    ├── Non-English characters → REJECTED (logged)
    └── English/Latin → Continue
```

### **Phase 2: Content Extraction by File Type**

#### **Text Files (.txt, .txt.utf-8, etc.)**
```
📄 Text File
    ↓
📖 Read with UTF-8 encoding (errors='ignore')
    ↓
🧹 clean_gutenberg_text()
    ├── Remove Project Gutenberg headers/footers
    ├── Remove metadata patterns
    └── Clean whitespace
```

#### **PDF Files**
```
📄 PDF File
    ↓
🔧 extract_text_from_pdf()
    ├── Try system pdftotext (preferred)
    └── Fallback to PyPDF2
    ↓
🧹 clean_pdf_text()
    ├── Remove page numbers
    ├── Remove library stamps
    ├── Remove headers/footers
    └── Fix OCR artifacts (0→O, 1→I, 5→S, 8→B, rn→m, cl→d, etc.)
```

#### **HTML Files**
```
📄 HTML File
    ↓
🔧 clean_html_text()
    ├── BeautifulSoup parsing (if available)
    ├── Remove script, style, nav, header, footer
    ├── Extract text content
    └── Remove wiki metadata
```

#### **XML Files (Old Bailey, London Lives)**
```
📄 XML File
    ↓
🔍 Detect XML Type
    ├── Old Bailey XML → extract_old_bailey_text()
    │   ├── Extract trial accounts
    │   ├── Extract front matter
    │   └── Preserve historical language
    └── London Lives XML → extract_london_lives_text()
        ├── Extract paragraphs with semantic markup
        ├── Extract lists
        └── Preserve person names, places, occupations
    ↓
🧹 Type-specific cleaning
    ├── Old Bailey → clean_old_bailey_text()
    └── London Lives → clean_london_lives_text()
```

### **Phase 3: Text Normalization**

```
📝 Extracted Text
    ↓
🔧 normalize_text()
    ├── Fix encoding issues (â€™→', â€œ→", â€"→—, etc.)
    ├── Normalize Unicode (NFC)
    ├── Handle single long lines → break_long_line()
    ├── Normalize line endings (\r\n→\n)
    └── Clean excessive whitespace
```

### **Phase 4: Quality Validation**

```
📝 Normalized Text
    ↓
🔍 Duplicate Detection
    ├── Content hash check → REJECTED if duplicate
    └── Continue if unique
    ↓
🌍 Language Detection
    ├── Non-English detected → REJECTED (logged)
    └── English → Continue
    ↓
📊 Quality Analysis (analyze_text_quality())
    ├── Length checks (min 200 chars)
    ├── Project Gutenberg validation (relaxed criteria)
    ├── Historical text validation (very relaxed)
    ├── OCR artifact detection
    ├── Advertisement density check
    └── Meaningful word ratio (≥50%)
    ↓
❌ Poor Quality → REJECTED (logged with details)
✅ Good Quality → Continue
```

### **Phase 5: Final Processing**

```
✅ Validated Text
    ↓
💾 Save to Processed Directory
    ├── Filename: cleaned_{original_stem}.txt
    ├── UTF-8 encoding
    └── Update statistics
    ↓
📊 Statistics Tracking
    ├── Characters before/after
    ├── Files processed/cleaned/skipped/failed
    ├── OCR artifacts fixed
    ├── Gutenberg headers removed
    └── Rejection reasons logged
```

## 🔧 **Detailed Cleaning Functions**

### **Project Gutenberg Cleaning**
```python
def clean_gutenberg_text(text: str) -> str:
    # Remove START/END markers
    # Remove metadata patterns (Title, Author, Release Date, etc.)
    # Clean excessive whitespace
    # Preserve historical language patterns
```

### **PDF Cleaning**
```python
def clean_pdf_text(text: str) -> str:
    # Remove page numbers: [Page 123], Page 123, standalone numbers
    # Remove library stamps: Internet Archive, Google, etc.
    # Remove headers/footers: all-caps lines, chapter numbers
    # Fix OCR errors: 0→O, 1→I, 5→S, 8→B, rn→m, cl→d, ii→n, vv→w, ſ→s
```

### **HTML Cleaning**
```python
def clean_html_text(html_content: str) -> str:
    # BeautifulSoup parsing (if available)
    # Remove unwanted elements: script, style, nav, header, footer, aside
    # Extract text content with proper spacing
    # Remove wiki metadata: "This page was last edited", "Jump to navigation", etc.
```

### **XML Cleaning (Old Bailey)**
```python
def extract_old_bailey_text(soup) -> str:
    # Extract trial accounts (main narrative)
    # Extract front matter (session info)
    # Preserve historical language and structure
    # Clean up XML artifacts while maintaining readability
```

### **XML Cleaning (London Lives)**
```python
def extract_london_lives_text(soup) -> str:
    # Extract paragraphs with semantic markup
    # Preserve person names, places, occupations, dates
    # Extract lists with proper formatting
    # Clean up markup artifacts
```

## 📈 **Quality Analysis Details**

### **OCR Artifact Detection**
- **Long capitals**: `[A-Z]{5,}\s+[A-Z]{5,}`
- **Spaced letters**: `\b[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\b`
- **Special characters**: `[!@#$%^&*()]{3,}`
- **Mixed numbers/letters**: `\b\d+[A-Za-z]+\d+\b`
- **Long non-word sequences**: `[^\w\s]{10,}`

### **Advertisement Detection**
- **Common patterns**: "this day is published", "just ready", "elegantly bound"
- **Price notations**: "price \d+s"
- **Publisher references**: "paternoster row", "corner of", "publishers"
- **Book advertisements**: "now ready", "new novels", "advertisements"

### **Meaningful Word Analysis**
- **Ratio calculation**: meaningful_words / total_words
- **Threshold**: ≥50% for general content
- **Relaxed criteria**: ≥40% for Project Gutenberg
- **Very relaxed**: ≥1000 chars + 100 words for historical texts

## 🚫 **Rejection Reasons & Logging**

### **Automatic Rejections**
1. **Non-English filename** (Arabic, Chinese, Cyrillic, etc.)
2. **Duplicate content** (hash-based detection)
3. **Non-English content** (language detection)
4. **Poor quality** (OCR artifacts, ads, low meaningful word ratio)
5. **Too short** (<200 characters, <5 lines, <50 words)
6. **Unsupported file type** (binary, unknown format)

### **Rejection Logging**
```json
{
  "timestamp": "2024-01-15 10:30:45",
  "file_path": "/path/to/file.txt",
  "filename": "file.txt",
  "file_size": 1234,
  "rejection_reason": "Poor content quality",
  "details": {
    "text_length": 150,
    "rejection_reasons": ["Text too short (< 200 chars)"],
    "ocr_issues": [],
    "advertisement_indicators": [],
    "meaningful_word_ratio": 0.3
  },
  "preview": "First 500 characters of file content"
}
```

## 📊 **Statistics Tracking**

### **Processing Statistics**
- **Files**: downloaded, processed, cleaned, skipped, failed, sanitized
- **Characters**: before cleaning, after cleaning, removed
- **Quality**: Gutenberg headers removed, OCR artifacts fixed, HTML markup removed
- **Content**: duplicates found, non-English skipped, poor quality skipped
- **Sources**: Gutenberg processed/accepted, encoding issues fixed

### **Rejection Statistics**
- **Total rejected files** with detailed reasons
- **Rejection summary** by reason type
- **File previews** for manual review
- **Quality analysis details** for each rejection

## 🔄 **Corpus Creation Process**

```
📁 Cleaned Files
    ↓
🔧 create_comprehensive_corpus()
    ├── Read all cleaned_*.txt files
    ├── Split into training segments (split_into_training_segments)
    │   ├── Split on double newlines (paragraphs)
    │   ├── Max length: 2000 characters
    │   ├── Min length: 100 characters
    │   └── Further split long segments at sentence boundaries
    ├── Filter segments (min 50 characters)
    └── Write to london_historical_corpus_comprehensive.txt
```

## 🎯 **Key Differences from TEXT_CLEANING_GUIDE.md**

### **What's Actually Implemented**
- ✅ **File type detection** and routing
- ✅ **Format-specific cleaning** (PDF, HTML, XML)
- ✅ **Historical text preservation** (Old Bailey, London Lives)
- ✅ **Quality analysis** with detailed rejection logging
- ✅ **Duplicate detection** using content hashing
- ✅ **Language detection** and filtering
- ✅ **OCR artifact detection** and correction
- ✅ **Advertisement detection** and filtering
- ✅ **Corpus segmentation** for training

### **What's Not Implemented**
- ❌ **Separate text_cleaner.py** script (functionality integrated)
- ❌ **Quality scoring system** (0-100 points)
- ❌ **Batch processing** with separate scripts
- ❌ **Advanced OCR correction** patterns
- ❌ **Separate analysis tools**

## 🔗 **Integration Points**

### **Called From**
- `historical_data_collector.py` → `process_file()` method
- `download_and_process_sources()` → Main collection pipeline
- `create_comprehensive_corpus()` → Final corpus creation

### **Configuration**
- **Data source flags** in `config.py` (enable_old_bailey, enable_london_lives, etc.)
- **Quality thresholds** hardcoded in analysis functions
- **File type detection** based on extensions and content

---

**This is the actual cleaning process implemented in the codebase, not the theoretical process described in TEXT_CLEANING_GUIDE.md.**
