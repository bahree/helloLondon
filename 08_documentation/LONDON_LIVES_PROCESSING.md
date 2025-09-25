# London Lives XML Processing Guide

## Overview

This guide covers the specialized processing of London Lives XML files for the London Historical LLM project. London Lives 1690-1800 contains over 1,600 digitized documents from 18th-century London institutions, providing rich historical data for training language models on historical London English.

## What is London Lives?

London Lives is a digital resource based on 18th-century institutional, administrative manuscript sources from London. The project prioritizes non-elite individuals named in these sources, facilitating searching for people and tracing their lives and experiences. The data emphasizes semantic markup of names, places, dates, and occupations.

## XML Structure

### Key Elements
- **Names**: `<name>` tags with attributes for given name, surname, sex, and unique IDs
- **Places**: `<geo>` tags for geographic information
- **Dates**: `<date>` tags for temporal information
- **Occupations**: `<occupation>` tags for professional information
- **Lists**: `<list>` and `<item>` tags for structured data

### Document Types
The dataset includes 59 different document types across 14 archives:

#### Major Categories
- **Parish Records**: Account books, payment records, workhouse registers
- **Legal Documents**: Sessions papers, criminal registers, inquests
- **Apprenticeship Records**: Indentures, registers, disciplinary cases
- **Poor Relief**: Settlement examinations, bastardy bonds, pauper lists
- **Hospital Records**: Admission/discharge registers, minute books

## File Organization

### Directory Structure
```
data/london_historical/manual_downloads/
├── london_lives/
│   ├── [all XML files in single folder]
│   └── processed/
├── old_bailey/
├── xml_files/
├── html_files/
└── pdf_files/
```

### File Classification
Files are organized by document type code extracted from filename:
- **EP**: Pauper Settlement and Bastardy Examinations
- **CR**: Criminal Registers of Prisoners
- **PS**: Sessions Papers - Justices Working Documents
- **AC**: Churchwardens and Overseers Account Books
- **And 55+ other types...**

## Processing Pipeline

### 1. File Transfer

#### From Windows to Ubuntu Server
```bash
# Transfer entire directory
scp -r C:\temp\data\london_lives\ amit@slmlocal1.guest.corp.microsoft.com:/home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/

# Transfer with compression and progress
scp -r -C -v C:\temp\data\london_lives\ amit@slmlocal1.guest.corp.microsoft.com:/home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/

# Transfer individual files
scp C:\temp\data\london_lives\*.xml amit@slmlocal1.guest.corp.microsoft.com:/home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/
```

#### Alternative: Using WSL
```bash
wsl scp -r /mnt/c/temp/data/london_lives/ amit@slmlocal1.guest.corp.microsoft.com:/home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/
```

### 2. File Organization

#### Automatic Organization
```bash
# Use the specialized organizer
cd /home/amit/src/helloLondon/02_data_collection/
python3 organize_london_lives.py /path/to/london_lives/xml/files --process
```

#### Manual Organization
```bash
# Create single directory for all London Lives files
mkdir -p /home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/

# Copy all XML files to the single folder
cp /path/to/*.xml /home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/
```

### 3. Text Extraction

#### Specialized London Lives Processing
The system automatically detects London Lives XML files and extracts:

- **Person Names**: Tagged with given name, surname, sex, and unique IDs
- **Geographic Information**: Places with type classification
- **Occupations**: Professional information
- **Dates**: Temporal information with type classification
- **Document Metadata**: Titles, types, and structural information

#### Example Output
```
DOCUMENT: Pauper Settlement Examination - GLBAEP10315
TYPE: examination

This is the examination of [PERSON: Ann Ross (Given: ANN, Surname: ROSS, Sex: f)], a [OCCUPATION: spinner] of [PLACE: St Botolph Aldgate (Type: parish)], taken on [DATE: 15th January 1734 (Type: examination)].

The said [PERSON: Ann Ross (Given: ANN, Surname: ROSS, Sex: f)] was born in [PLACE: London] and has lived in [PLACE: St Botolph Aldgate (Type: parish)] for the past five years. She is the wife of [PERSON: John Ross (Given: JOHN, Surname: ROSS, Sex: m)], a [OCCUPATION: weaver].

• [PERSON: Mary Smith (Given: MARY, Surname: SMITH, Sex: f)], [OCCUPATION: midwife] of [PLACE: St Botolph Aldgate (Type: parish)]
• [PERSON: Thomas Jones (Given: THOMAS, Surname: JONES, Sex: m)], [OCCUPATION: churchwarden] of [PLACE: St Botolph Aldgate (Type: parish)]
```

### 4. Processing Commands

#### Test Processing
```bash
# Test with sample data
cd /home/amit/src/helloLondon/02_data_collection/
python3 test_london_lives_processing.py

# Test with real file
python3 test_london_lives_processing.py /path/to/sample_london_lives.xml
```

#### Full Processing
```bash
# Process all files including London Lives XML
cd /home/amit/src/helloLondon/02_data_collection/
python3 historical_data_collector.py

# Or process just London Lives files
python3 organize_london_lives.py /path/to/london_lives/xml/files --process
```

## Technical Implementation

### Key Components

#### 1. London Lives Detection
```python
def is_london_lives_xml(self, soup) -> bool:
    """Check if this is a London Lives XML file"""
    # Check for name elements with London Lives ID pattern (nX-Y)
    name_elements = soup.find_all('name')
    if name_elements:
        for name in name_elements[:5]:
            name_id = name.get('id', '')
            if name_id and name_id.startswith('n') and '-' in name_id:
                return True
    
    # Check for semantic markup elements
    geo_elements = soup.find_all('geo')
    occupation_elements = soup.find_all('occupation')
    date_elements = soup.find_all('date')
    
    if len(geo_elements) > 0 and len(occupation_elements) > 0 and len(date_elements) > 0:
        return True
        
    return False
```

#### 2. Text Extraction
```python
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
    
    return '\n\n'.join(extracted_text)
```

#### 3. Semantic Markup Processing
```python
def extract_london_lives_paragraph(self, paragraph) -> str:
    """Extract paragraph text with London Lives semantic markup"""
    text_parts = []
    
    for element in paragraph.descendants:
        if element.name == 'name':
            # Extract person name with attributes
            name_text = element.get_text().strip()
            given = element.get('given', '')
            surname = element.get('surname', '')
            sex = element.get('sex', '')
            
            # Format with context
            name_parts = []
            if given: name_parts.append(f"Given: {given}")
            if surname: name_parts.append(f"Surname: {surname}")
            if sex: name_parts.append(f"Sex: {sex}")
            
            if name_parts:
                text_parts.append(f"[PERSON: {name_text} ({', '.join(name_parts)})]")
            else:
                text_parts.append(f"[PERSON: {name_text}]")
                
        elif element.name == 'geo':
            # Extract geographic information
            geo_text = element.get_text().strip()
            geo_type = element.get('type', '')
            if geo_type:
                text_parts.append(f"[PLACE: {geo_text} (Type: {geo_type})]")
            else:
                text_parts.append(f"[PLACE: {geo_text}]")
                
        # ... handle other elements
```

## Data Quality and Statistics

### Expected Output
- **Person Names**: Properly tagged with given name, surname, sex, and unique IDs
- **Geographic Information**: Places with type classification (parish, place, etc.)
- **Occupations**: Professional information for historical context
- **Dates**: Temporal information with type classification
- **Document Structure**: Organized by document type and archive

### Quality Metrics
- **Semantic Markup**: Names, places, dates, and occupations properly tagged
- **Content Preservation**: All relevant historical text extracted
- **Structure Maintenance**: Document narratives maintain logical flow
- **Metadata Retention**: Important contextual information preserved

## Document Type Reference

### Major Document Types
- **EP**: Pauper Settlement and Bastardy Examinations
- **CR**: Criminal Registers of Prisoners
- **PS**: Sessions Papers - Justices Working Documents
- **AC**: Churchwardens and Overseers Account Books
- **IA**: Apprenticeship Indentures and Disciplinary Cases
- **IC**: Coroners Inquests into Suspicious Deaths
- **RH**: St Thomas Admission and Discharge Registers
- **MV**: Minutes of Parish Vestries

### Archive Codes
- **GL**: Guildhall Library
- **LM**: London Metropolitan Archives
- **NA**: The National Archives
- **WA**: Westminster Abbey Muniment Room
- **WC**: Westminster Archives Centre

## Simplified Single Folder Approach

**Important**: This implementation uses a simplified single folder approach instead of organizing files into 59 separate subfolders by document type. All London Lives XML files are placed in one folder (`london_lives/`) for easier management while maintaining full processing capabilities.

### Benefits of Single Folder Approach
- **Simpler file management**: No need to organize files into 59 different subfolders
- **Easier transfers**: All files go to one location
- **Faster processing**: No complex folder traversal needed
- **Same functionality**: All semantic markup and processing features preserved

## Troubleshooting

### Common Issues

#### 1. File Transfer Problems
```bash
# Test SSH connection first
ssh amit@slmlocal1.guest.corp.microsoft.com

# Check target directory exists
mkdir -p /home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/
```

#### 2. Processing Errors
```bash
# Check file permissions
chmod 755 /home/amit/src/helloLondon/data/london_historical/manual_downloads/london_lives/

# Verify XML file integrity
file /path/to/london_lives.xml
```

#### 3. Document Type Classification
```bash
# Test document type extraction
python3 test_london_lives_processing.py
```

## Best Practices

### 1. File Organization
- Organize files by document type for easier management
- Keep original files in a separate directory
- Use descriptive folder names based on document types

### 2. Processing
- Test with a small sample first
- Monitor processing logs for errors
- Verify output quality before full processing

### 3. Quality Control
- Check extracted text for completeness
- Verify semantic markup is properly preserved
- Ensure document narratives are coherent

## Integration with Main Pipeline

The London Lives processing is fully integrated with the main data collection pipeline:

1. **Automatic Detection**: London Lives XML files are automatically identified
2. **Specialized Processing**: Custom extraction logic for semantic markup
3. **Corpus Integration**: Extracted text is added to the comprehensive corpus
4. **Statistics Tracking**: Processing statistics include London Lives-specific metrics

## Future Enhancements

### Potential Improvements
- **Social Network Analysis**: Extract relationships between people
- **Geographic Analysis**: Map locations and analyze spatial patterns
- **Temporal Analysis**: Better date extraction and normalization
- **Demographic Analysis**: Extract information about age, gender, occupation
- **Institutional Analysis**: Study patterns across different document types

### Research Applications
- **Social History**: Analyze poor relief, apprenticeship, and criminal justice
- **Demographic History**: Study population movements and social structures
- **Economic History**: Analyze occupations and economic activities
- **Legal History**: Study legal proceedings and outcomes
- **Urban History**: Analyze London's social and economic development

## References

- [London Lives 1690-1800](http://www.londonlives.org/)
- [Project Background](https://www.londonlives.org/static/Background.jsp)
- [Document Types](https://www.londonlives.org/static/Documents.jsp)
- [Technical Methods](https://www.londonlives.org/static/Project.jsp#toc7)
- [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/)

## License and Usage

This processing system is designed for research and educational purposes. When using London Lives data, please:

1. **Cite the Source**: Reference London Lives 1690-1800 in your work
2. **Respect Copyright**: Follow the Creative Commons Attribution-NonCommercial 4.0 License
3. **Maintain Attribution**: Preserve original metadata and structure
4. **Share Responsibly**: Consider privacy implications of historical data

---

*This documentation is part of the London Historical LLM project. For questions or contributions, please contact the project maintainers.*
