# Old Bailey XML Processing Guide

## Overview

This guide covers the specialized processing of Old Bailey XML files for the London Historical LLM project. The Old Bailey Online (OBP) contains over 197,000 criminal trials from 1674-1913, providing rich historical data for training language models on historical London English.

> **⚠️ Note**: Old Bailey processing is currently **disabled by default** in the configuration. To enable it, set `"enable_old_bailey": True` in `config.py` line 160.

## What is Old Bailey?

The Old Bailey was London's central criminal court from 1674-1913. The Old Bailey Online project has digitized and marked up these records in TEI XML format, making them available for historical research and computational analysis.

## XML Structure

### Basic Structure
```xml
<TEI.2>
<text>
<body>
<div0 type="sessionsPaper">
<div1 type="frontMatter">[title pages and lists of jurors]</div1>
<div1 type="trialAccount">[one per trial]</div1>
<div1 type="supplementaryMaterial">[outcomes and sentences]</div1>
<div1 type="punishmentSummary">[sentencing details]</div1>
<div1 type="advertisements">[part 1 only]</div1>
</div0>
</body>
</text>
</TEI.2>
```

### Key Elements
- **Sessions Papers**: Trial accounts with detailed proceedings
- **Ordinary's Accounts**: Execution reports and biographical details
- **Person Names**: Marked with `<persName>` tags including type (defendant, victim, witness, etc.)
- **Offences**: Marked with `<rs>` tags with categories and subcategories
- **Verdicts and Sentences**: Linked to defendants via `<join>` tags

## File Organization

### Directory Structure
```
data/london_historical/manual_downloads/
├── old_bailey/
│   ├── sessions_papers/          # Trial accounts (main content)
│   ├── ordinarys_accounts/       # Execution reports
│   ├── other_xml/               # Other XML files
│   └── processed/               # Processed files
└── [other_data_sources]/        # Other historical data sources
```

### File Classification
- **Sessions Papers**: Files containing `type="sessionsPaper"`
- **Ordinary's Accounts**: Files containing `type="ordinarysAccount"`
- **Other XML**: Files that don't match the above patterns

## Processing Pipeline

### 1. File Organization

#### Automatic Organization
```bash
# Use the specialized organizer (if available)
cd 02_data_collection/
python3 organize_old_bailey.py /path/to/old_bailey/xml/files --process
```

#### Manual Organization
```bash
# Create directory structure
mkdir -p data/london_historical/manual_downloads/old_bailey/
mkdir -p data/london_historical/manual_downloads/old_bailey/sessions_papers/
mkdir -p data/london_historical/manual_downloads/old_bailey/ordinarys_accounts/
mkdir -p data/london_historical/manual_downloads/old_bailey/other_xml/

# Copy files to appropriate folders
cp /path/to/sessions_papers/*.xml data/london_historical/manual_downloads/old_bailey/sessions_papers/
cp /path/to/ordinarys_accounts/*.xml data/london_historical/manual_downloads/old_bailey/ordinarys_accounts/
```

### 2. Text Extraction

#### Specialized Old Bailey Processing
The system automatically detects Old Bailey XML files and extracts:

- **Trial Accounts**: Main narrative content with proper formatting
- **Person Names**: Tagged with roles (DEFENDANT, VICTIM, WITNESS, JUROR, etc.)
- **Front Matter**: Titles, juror lists, and administrative information
- **Outcomes**: Verdicts and sentencing information
- **Punishments**: Detailed sentencing information

#### Example Output
```
Session of 1834-01-15
Trial: JOHN DOE, Theft > burglary, 15th January 1834
The prisoner was indicted for burglariously breaking and entering the dwelling-house of Samuel Raley on the 7th of December.
The witness Mary Smith testified that she saw the prisoner at the scene of the crime.
The jury found the prisoner guilty.
The prisoner was found guilty and sentenced to transportation for seven years.
```

### 3. Processing Commands

#### Test Processing
```bash
# Test with sample data
cd 02_data_collection/
python3 test_old_bailey_processing.py

# Test with real file
python3 test_old_bailey_processing.py /path/to/sample_old_bailey.xml
```

#### Full Processing
```bash
# Process all files including Old Bailey XML (XML files only, skips ZIP)
cd 02_data_collection/
python3 historical_data_collector.py
```

**Note**: The system now automatically:
- **Only processes XML files** from Old Bailey directories
- **Skips ZIP files** and other non-XML formats
- **Uses configuration from config.py** (no command-line overrides needed)
- **Preserves historical language** with minimal cleaning
- **Requires explicit enablement** via `"enable_old_bailey": True` in config.py

## Technical Implementation

### Key Components

#### 1. Old Bailey Detection
```python
def is_old_bailey_xml(self, soup) -> bool:
    """Check if this is an Old Bailey XML file"""
    tei_elements = soup.find_all('TEI.2')
    if tei_elements:
        sessions_paper = soup.find('div0', {'type': 'sessionsPaper'})
        ordinarys_account = soup.find('div0', {'type': 'ordinarysAccount'})
        return sessions_paper is not None or ordinarys_account is not None
    return False
```

#### 2. Text Extraction
```python
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
    
    return combined_text.strip()
```

#### 3. Trial Narrative Extraction
```python
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
```

#### 4. Paragraph Processing
```python
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
```

## Data Quality and Statistics

### Expected Output
- **Sessions Papers**: Detailed trial narratives with natural historical language
- **Ordinary's Accounts**: Biographical details and execution reports
- **Narrative Text**: Preserved historical spelling and punctuation
- **Legal Proceedings**: Verdicts, sentences, and outcomes
- **Historical Context**: Dates, locations, and social context
- **Clean Text**: Minimal processing to maintain historical authenticity

### Quality Metrics
- **Historical Language Preservation**: Original spelling, punctuation, and grammar maintained
- **Content Preservation**: All relevant historical text extracted
- **Structure Maintenance**: Trial narratives maintain logical flow
- **Metadata Retention**: Important contextual information preserved
- **XML-Only Processing**: Only processes XML files, skips ZIP and other formats
- **Minimal Cleaning**: Preserves historical authenticity while removing artifacts

## Troubleshooting

### Common Issues

#### 1. File Organization Problems
```bash
# Check target directory exists
mkdir -p data/london_historical/manual_downloads/old_bailey/

# Verify directory structure
ls -la data/london_historical/manual_downloads/old_bailey/
```

#### 2. Processing Errors
```bash
# Check file permissions
chmod 755 data/london_historical/manual_downloads/old_bailey/

# Verify XML file integrity
file /path/to/old_bailey.xml
```

#### 3. Memory Issues with Large Files
```bash
# Process files in batches
find /path/to/old_bailey/ -name "*.xml" | head -100 | xargs -I {} python3 process_single_file.py {}
```

## Best Practices

### 1. File Organization
- Organize files by type (sessions_papers, ordinarys_accounts)
- Keep original files in a separate directory
- Use descriptive filenames when possible

### 2. Processing
- Test with a small sample first
- Monitor processing logs for errors
- Verify output quality before full processing

### 3. Quality Control
- Check extracted text for completeness
- Verify person name tagging
- Ensure trial narratives are coherent

## Integration with Main Pipeline

The Old Bailey processing is fully integrated with the main data collection pipeline:

1. **Automatic Detection**: Old Bailey XML files are automatically identified by TEI structure
2. **XML-Only Processing**: Only processes `.xml` files, skips ZIP and other formats
3. **Configuration-Driven**: Uses settings from `config.py` (no command-line overrides needed)
4. **Specialized Processing**: Custom extraction logic for historical content
5. **Corpus Integration**: Extracted text is added to the comprehensive corpus
6. **Statistics Tracking**: Processing statistics include Old Bailey-specific metrics
7. **Historical Language Preservation**: Minimal cleaning to maintain authenticity
8. **Optional Feature**: Must be explicitly enabled via `"enable_old_bailey": True` in config.py

### Enabling Old Bailey Processing

To enable Old Bailey XML processing:

1. **Edit Configuration**: Set `"enable_old_bailey": True` in `config.py` line 160
2. **Prepare Data**: Place Old Bailey XML files in `data/london_historical/manual_downloads/old_bailey/`
3. **Run Collection**: Execute `python3 02_data_collection/historical_data_collector.py`
4. **Verify Processing**: Check logs for Old Bailey-specific processing messages

### Data Preparation

Before processing, organize your Old Bailey XML files:

```bash
# Create the directory structure
mkdir -p data/london_historical/manual_downloads/old_bailey/sessions_papers/
mkdir -p data/london_historical/manual_downloads/old_bailey/ordinarys_accounts/
mkdir -p data/london_historical/manual_downloads/old_bailey/other_xml/

# Copy your XML files to the appropriate directories
# Sessions papers go in sessions_papers/
# Ordinary's accounts go in ordinarys_accounts/
# Other XML files go in other_xml/
```

## Future Enhancements

### Potential Improvements
- **Geographic Tagging**: Extract and tag London locations
- **Temporal Analysis**: Better date extraction and normalization
- **Social Network Analysis**: Extract relationships between people
- **Crime Classification**: Automatic categorization of offences
- **Sentiment Analysis**: Analyze emotional content of trials

### Research Applications
- **Historical Linguistics**: Study language change over time
- **Social History**: Analyze crime patterns and social structures
- **Legal History**: Study legal proceedings and outcomes
- **Demographic Analysis**: Extract information about age, gender, occupation

## References

- [Old Bailey Online](https://www.oldbaileyonline.org/)
- [TEI Guidelines](https://tei-c.org/guidelines/)
- [Old Bailey XML Schema](https://www.oldbaileyonline.org/static/Documentation.jsp)
- [Historical Data Processing](https://github.com/bahree/helloLondon/tree/main/02_data_collection)

## License and Usage

This processing system is designed for research and educational purposes. When using Old Bailey data, please:

1. **Cite the Source**: Reference Old Bailey Online in your work
2. **Respect Copyright**: Follow Old Bailey Online's terms of use
3. **Maintain Attribution**: Preserve original metadata and structure
4. **Share Responsibly**: Consider privacy implications of historical data

---

*This documentation is part of the London Historical LLM project. For questions or contributions, please contact the project maintainers.*
