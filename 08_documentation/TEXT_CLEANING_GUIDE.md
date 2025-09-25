# 🧹 Text Cleaning Guide for Historical Documents

This guide explains the comprehensive text cleaning system for historical documents, including OCR artifact removal, encoding fixes, and formatting normalization.

## 🎯 Overview

Historical documents often contain quality issues that can affect model training:
- **OCR Artifacts**: Character substitutions, spacing issues, broken words
- **Encoding Problems**: Smart quotes, special characters, mojibake
- **Formatting Issues**: Excessive whitespace, page numbers, headers
- **Content Quality**: Repetitive text, missing punctuation, very short content

## 🔧 Text Cleaning System

### **Components**

1. **`text_cleaner.py`** - Advanced text cleaning engine
2. **`text_quality_analyzer.py`** - Quality analysis and recommendations
3. **`enhanced_data_downloader.py`** - Integrated download and cleaning
4. **`run_text_cleaning.py`** - Master script for complete cleaning

### **Features**

- **OCR Artifact Removal**: Fixes common OCR character substitutions
- **Encoding Normalization**: Handles smart quotes, special characters, mojibake
- **Formatting Cleanup**: Removes excessive whitespace, page numbers, headers
- **Historical Text Preservation**: Maintains historical language patterns
- **Quality Analysis**: Provides detailed quality metrics and recommendations

## 🚀 Quick Start

### **Option 1: Complete Cleaning Process**
```bash
cd 02_data_collection
python3 run_text_cleaning.py
```

### **Option 2: Step-by-Step Cleaning**
```bash
# 1. Analyze text quality
python3 text_quality_analyzer.py

# 2. Clean text files
python3 text_cleaner.py

# 3. Verify results
python3 text_quality_analyzer.py --input_dir data/london_historical/cleaned
```

### **Option 3: Integrated Download and Cleaning**
```bash
# Download and clean in one step
python3 enhanced_data_downloader.py

# Or disable cleaning
python3 enhanced_data_downloader.py --no-cleaning
```

## 📊 Quality Analysis

### **Quality Metrics**

The system analyzes text quality using multiple metrics:

#### **Encoding Issues (5 points each)**
- Smart quotes encoding (`â€™` → `'`)
- Em dash encoding (`â€"` → `—`)
- En dash encoding (`â€"` → `–`)
- Bullet point encoding (`â€¢` → `•`)
- Ellipsis encoding (`â€¦` → `…`)
- Mojibake detection

#### **OCR Artifacts (3 points each)**
- Numbers in words (common OCR error)
- Mixed alphanumeric patterns
- Character substitutions (`rn` → `m`, `cl` → `d`, etc.)
- Spacing issues

#### **Formatting Issues (2 points each)**
- Excessive newlines
- Excessive spaces
- Blank lines
- Broken words from OCR
- Broken sentences from OCR
- Page numbers
- Headers

#### **Content Quality (4 points each)**
- Very short content
- Repetitive content
- Lack of punctuation
- Excessive special characters

### **Quality Score Calculation**

- **Starting Score**: 100 points
- **Deductions**: Based on issue type and severity
- **Final Score**: 0-100 (higher is better)

### **Quality Categories**

- **Excellent (90-100)**: High-quality text, minimal issues
- **Good (70-89)**: Good quality with some minor issues
- **Fair (50-69)**: Moderate quality with noticeable issues
- **Poor (<50)**: Low quality with significant issues

## 🧹 Cleaning Process

### **Step 1: Encoding Fixes**

```python
# Fix smart quotes
text = text.replace('â€™', "'")
text = text.replace('â€œ', '"')
text = text.replace('â€', '"')

# Fix dashes
text = text.replace('â€"', '—')
text = text.replace('â€"', '–')

# Fix other characters
text = text.replace('â€¢', '•')
text = text.replace('â€¦', '…')
```

### **Step 2: OCR Artifact Removal**

```python
# Common OCR corrections
ocr_corrections = {
    r'\b0\b': 'O',  # Zero to O in words
    r'\b1\b': 'I',  # One to I in words
    r'\b5\b': 'S',  # Five to S in words
    r'\b8\b': 'B',  # Eight to B in words
    r'rn': 'm',    # rn to m
    r'cl': 'd',    # cl to d
    r'ii': 'n',    # ii to n
    r'vv': 'w',    # vv to w
}
```

### **Step 3: Formatting Cleanup**

```python
# Remove excessive whitespace
text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
text = re.sub(r'[ \t]{4,}', ' ', text)

# Fix broken words
text = re.sub(r'([a-zA-Z])\n([a-zA-Z])', r'\1\2', text)

# Remove page numbers
text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

# Remove headers
text = re.sub(r'^\s*[A-Z][A-Z\s]+\s*$', '', text, flags=re.MULTILINE)
```

### **Step 4: Historical Text Normalization**

```python
# Normalize Unicode
text = unicodedata.normalize('NFKC', text)

# Fix historical characters
text = re.sub(r'ſ', 's', text)  # Long s to regular s

# Preserve historical punctuation
text = re.sub(r'\.{3,}', '…', text)  # Multiple dots to ellipsis
text = re.sub(r'-{2,}', '—', text)   # Multiple dashes to em dash
```

## 📁 Output Structure

```
data/london_historical/
├── [original files]
├── cleaned/
│   ├── cleaned_file1.txt
│   ├── cleaned_file2.txt
│   ├── issues_file1.json
│   ├── issues_file2.json
│   ├── cleaning_statistics.json
│   └── text_cleaning.log
├── analysis/
│   ├── detailed_analyses.json
│   ├── quality_summary.json
│   └── quality_analysis.log
└── london_historical_corpus_cleaned.txt
```

### **File Contents**

- **`cleaned_*.txt`**: Cleaned text files
- **`issues_*.json`**: Detailed issue reports for each file
- **`cleaning_statistics.json`**: Overall cleaning statistics
- **`detailed_analyses.json`**: Detailed quality analysis
- **`quality_summary.json`**: Summary of quality metrics

## 🔍 Issue Detection

### **Encoding Issues**

The system detects and fixes:
- **Smart Quotes**: `â€™` → `'`, `â€œ` → `"`, `â€` → `"`
- **Dashes**: `â€"` → `—`, `â€"` → `–`
- **Special Characters**: `â€¢` → `•`, `â€¦` → `…`
- **Mojibake**: Garbled text from encoding errors

### **OCR Artifacts**

Common OCR issues fixed:
- **Character Substitutions**: `rn` → `m`, `cl` → `d`, `ii` → `n`, `vv` → `w`
- **Number Substitutions**: `0` → `O`, `1` → `I`, `5` → `S`, `8` → `B`
- **Spacing Issues**: Broken words and sentences
- **Mixed Alphanumeric**: OCR artifacts in text

### **Formatting Issues**

Formatting problems addressed:
- **Excessive Whitespace**: Multiple newlines, spaces, tabs
- **Page Numbers**: Standalone numbers on lines
- **Headers**: All-caps lines
- **Blank Lines**: Empty or whitespace-only lines

## 📈 Quality Improvement

### **Before Cleaning**

Typical issues in historical documents:
- **Encoding Problems**: `â€™`, `â€œ`, `â€"`, `â€"`, `â€¢`, `â€¦`
- **OCR Artifacts**: `rn` → `m`, `cl` → `d`, numbers in words
- **Formatting Issues**: Excessive whitespace, page numbers, headers
- **Content Quality**: Repetitive text, missing punctuation

### **After Cleaning**

Improved text quality:
- **Clean Encoding**: Proper Unicode characters
- **Fixed OCR**: Corrected character substitutions
- **Normalized Formatting**: Consistent spacing and structure
- **Preserved Content**: Historical language patterns maintained

## ⚙️ Configuration

### **Cleaning Options**

```python
# Enable/disable cleaning
enable_cleaning = True

# Custom OCR corrections
ocr_corrections = {
    r'\b0\b': 'O',
    r'\b1\b': 'I',
    # Add more as needed
}

# Quality thresholds
quality_thresholds = {
    'excellent': 90,
    'good': 70,
    'fair': 50,
    'poor': 0
}
```

### **Output Settings**

```python
# Output directories
cleaned_dir = "data/london_historical/cleaned"
analysis_dir = "data/london_historical/analysis"

# File naming
cleaned_prefix = "cleaned_"
issues_suffix = "_issues.json"
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Encoding Errors**
```bash
# Check file encoding
file -i filename.txt

# Convert encoding
iconv -f latin1 -t utf-8 input.txt > output.txt
```

#### **Memory Issues**
```bash
# Process files in smaller batches
python3 text_cleaner.py --batch_size 10
```

#### **Quality Analysis Fails**
```bash
# Check file permissions
ls -la data/london_historical/

# Verify file format
head -5 data/london_historical/*.txt
```

### **Getting Help**

1. **Check logs**: `tail -f data/london_historical/cleaned/text_cleaning.log`
2. **Review analysis**: Check `quality_summary.json` for issues
3. **Verify files**: Ensure input files are readable
4. **Check permissions**: Verify write access to output directories

## 📊 Monitoring and Maintenance

### **Regular Quality Checks**

```bash
# Analyze current data quality
python3 text_quality_analyzer.py

# Clean new data
python3 text_cleaner.py

# Verify improvements
python3 text_quality_analyzer.py --input_dir data/london_historical/cleaned
```

### **Quality Metrics Tracking**

- **Monitor quality scores** over time
- **Track common issues** and patterns
- **Update cleaning rules** based on new data
- **Maintain cleaning statistics** for reporting

## 🎯 Best Practices

### **Before Cleaning**

1. **Backup original data** before cleaning
2. **Analyze quality first** to understand issues
3. **Review cleaning rules** for your specific data
4. **Test on small samples** before full processing

### **During Cleaning**

1. **Monitor progress** through logs
2. **Check quality improvements** after each step
3. **Verify historical content** is preserved
4. **Handle errors gracefully** with fallbacks

### **After Cleaning**

1. **Compare before/after** quality metrics
2. **Review cleaned content** for accuracy
3. **Update training data** with cleaned versions
4. **Document cleaning process** for reproducibility

## 🔗 Related Documentation

- [Data Collection Guide](DATA_COLLECTION.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

---

**Ready to clean your historical documents for optimal model training!** 🧹📚

**Next Steps:**
1. Run quality analysis on your data
2. Apply text cleaning to improve quality
3. Verify improvements and update training data
4. Monitor quality metrics during training
