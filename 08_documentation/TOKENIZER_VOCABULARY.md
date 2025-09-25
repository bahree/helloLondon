# Hello London Tokenizer Vocabulary
*This vocabulary was designed specifically for the Hello London Historical LLM project, focusing on the period 1500-1850 and London-centric historical texts.*

This document describes the custom tokenizer vocabulary and special tokens designed specifically for historical London texts (1500-1850).

> **📊 Related Guide**: For practical token counting and dataset analysis, see [**TOKEN_COUNTING_GUIDE.md**](TOKEN_COUNTING_GUIDE.md) - the companion guide for measuring your dataset size and planning training resources.

## Overview

The Hello London tokenizer uses a **BPE (Byte Pair Encoding)** approach with **30,000 vocabulary size** and **150+ special tokens** optimized for historical English text. This design ensures efficient tokenization of archaic language, London-specific terms, and period-appropriate vocabulary.

## Special Token Categories

### 1. Basic Control Tokens
```
<|endoftext|>    - End of text marker
<|startoftext|>  - Start of text marker
<|pad|>          - Padding token
<|unk|>          - Unknown token
<|mask|>         - Masking token for training
<|sep|>          - Separator token
<|cls|>          - Classification token
<|eos|>          - End of sequence
<|bos|>          - Beginning of sequence
```

### 2. Historical Language Tokens
**Archaic Pronouns and Verbs:**
```
<|thou|>, <|thee|>, <|thy|>, <|thine|>  - Second person pronouns
<|hast|>, <|hath|>, <|doth|>, <|dost|>  - Archaic verb forms
<|art|>, <|wilt|>, <|shalt|>, <|canst|> - Modal verbs
```

**Adverbs and Expressions:**
```
<|verily|>, <|indeed|>, <|forsooth|>, <|methinks|>
<|perchance|>, <|anon|>, <|ere|>, <|whilst|>
<|betwixt|>, <|amongst|>, <|prithee|>, <|pray|>
<|beseech|>, <|ye|>, <|yon|>, <|fain|>
```

**Common Archaic Words:**
```
<|quoth|>, <|afeard|>, <|affright|>, <|albeit|>, <|howbeit|>
<|hither|>, <|thither|>, <|whence|>, <|whither|>, <|wherefore|>
<|hitherto|>, <|thereto|>, <|whereto|>, <|whereby|>
<|peradventure|>, <|truly|>, <|marry|>, <|goodmorrow|>, <|farewell|>
```

### 3. London-Specific Tokens
**General Locations:**
```
<|london|>, <|thames|>, <|westminster|>, <|city|>, <|borough|>
<|parish|>, <|ward|>", <|street|>, <|lane|>, <|court|>
```

**Establishments:**
```
<|tavern|>, <|inn|>, <|coffeehouse|>, <|market|>, <|fair|>
```

**Iconic London Landmarks:**
```
<|tower|>, <|stpauls|>, <|fleet|>, <|cheapside|>, <|smithfield|>
<|tyburn|>, <|newgate|>, <|southwark|>, <|coventgarden|>, <|billingsgate|>
<|leadenhall|>, <|guildhall|>, <|exchange|>, <|bridge|>, <|wharf|>
```

### 4. Historical Period Tokens
**Dynasties and Eras:**
```
<|tudor|>, <|stuart|>, <|georgian|>, <|regency|>, <|victorian|>
<|elizabethan|>, <|restoration|>, <|hanoverian|>, <|enlightenment|>
```

**Major Events:**
```
<|plague|>, <|fire|>, <|great|>, <|civil|>, <|war|>
<|gunpowder|>, <|popish|>, <|southsea|>, <|bubble|>, <|revolution|>, <|glorious|>
```

### 5. Social Class Tokens
**Nobility and Gentry:**
```
<|noble|>, <|gentleman|>, <|lady|>, <|yeoman|>, <|squire|>
<|knight|>, <|duke|>, <|earl|>
```

**Commoners and Trades:**
```
<|commoner|>, <|apprentice|>, <|servant|>, <|merchant|>, <|artisan|>
<|labourer|>, <|beggar|>, <|vagabond|>, <|pauper|>
```

**Civic Roles:**
```
<|alderman|>, <|burgess|>, <|freeman|>
```

### 6. Legal and Court Tokens
**Court Proceedings:**
```
<|trial|>, <|judge|>, <|jury|>, <|witness|>, <|accused|>
<|sentence|>, <|punishment|>, <|gaol|>, <|transport|>, <|hanging|>
```

**Legal Officials:**
```
<|magistrate|>, <|constable|>, <|watchman|>
```

**Punishments and Instruments:**
```
<|pillory|>, <|stocks|>, <|indictment|>, <|verdict|>, <|execution|>
<|tyburntree|>, <|newgatebird|>
```

### 7. Religious Tokens
**Institutions and Roles:**
```
<|church|>, <|parish|>, <|clergy|>, <|bishop|>, <|archbishop|>
<|puritan|>, <|dissenter|>, <|catholic|>, <|protestant|>, <|chapel|>
```

**Religious Practices:**
```
<|prayer|>, <|sermon|>, <|blessing|>, <|curse|>, <|sin|>
<|tithes|>, <|communion|>, <|heresy|>, <|papist|>, <|atheist|>
```

### 8. Economic Tokens
**Currency (Historical):**
```
<|shilling|>, <|pound|>, <|penny|>, <|guinea|>, <|crown|>
<|farthing|>, <|halfpenny|>, <|groat|>, <|sovereign|>, <|noble|>
```

**Trade and Commerce:**
```
<|trade|>, <|commerce|>, <|merchant|>, <|shop|>, <|warehouse|>
<|guild|>, <|livery|>, <|apprenticeship|>, <|bargain|>, <|usury|>
```

### 9. Time and Date Tokens
**Times of Day:**
```
<|morn|>, <|noon|>, <|eve|>, <|night|>, <|dawn|>, <|dusk|>
```

**Days of the Week:**
```
<|monday|>, <|tuesday|>, <|wednesday|>, <|thursday|>, <|friday|>
<|saturday|>, <|sunday|>
```

**Months:**
```
<|january|>, <|february|>, <|march|>, <|april|>, <|may|>, <|june|>
<|july|>, <|august|>, <|september|>, <|october|>, <|november|>, <|december|>
```

**Historical Time Periods:**
```
<|fortnight|>, <|sennight|>, <|michaelmas|>, <|ladyday|>, <|candlemas|>
<|midsummer|>, <|christmas|>, <|easter|>, <|whitsun|>, <|lent|>
```

### 10. Profession Tokens (New Category)
**Medical and Health:**
```
<|apothecary|>, <|barbersurgeon|>
```

**Transportation:**
```
<|coachman|>, <|linkboy|>, <|waterman|>
```

**Urban Services:**
```
<|chimneysweep|>, <|costermonger|>, <|nightsoilman|>
```

**Civic Roles:**
```
<|beadle|>, <|crier|>
```

### 11. Slang and Street Tokens (New Category)
**Living and Housing:**
```
<|doss|> (bed), <|ken|> (house)
```

**Money and Deception:**
```
<|fawney|> (ring), <|rig|> (trick), <|sup|> (drink)
```

**Appearance and Behavior:**
```
<|phiz|> (face), <|visage|>, <|countenance|>, <|mauther|> (girl)
<|brabble|> (quarrel), <|chuffed|> (pleased), <|bauchle|> (old shoe)
<|clomph|> (clumsy), <|cramboclink|> (nonsense), <|abroad|> (out)
```

## Design Rationale

### Historical Accuracy
- **Time Period**: Tokens span 1500-1850, covering Early Modern English through early Victorian era
- **Source Material**: Based on London Lives, Old Bailey records, Pepys' diary, and period literature
- **Language Evolution**: Captures both Elizabethan/Stuart archaic forms and emerging Victorian terms

### Efficiency Benefits
- **Single Token Treatment**: Common archaic words like "quoth" and "afeard" are single tokens
- **London Context**: Landmarks like "newgate" and "tyburn" don't get split into subwords
- **Period Recognition**: Historical events and periods are preserved as semantic units

### Vocabulary Size Management
- **Total Size**: 30,000 tokens (optimal for small language models)
- **Special Tokens**: ~150 tokens (0.5% of vocabulary)
- **BPE Learning**: Remaining tokens learned from training data
- **Frequency-Based**: Special tokens chosen for high frequency in historical texts

## Usage Examples

### Tokenization of Historical Text
```
Original: "Quoth the alderman, 'Tis a fair day at Newgate."
Tokens:   <|quoth|> the <|alderman|> , 'Tis a fair day at <|newgate|> .
```

### London-Specific Context
```
Original: "The Thames flowed past the Tower to Southwark."
Tokens:   The <|thames|> flowed past the <|tower|> to <|southwark|> .
```

### Archaic Language
```
Original: "Hast thou seen the apothecary's wares?"
Tokens:   <|hast|> <|thou|> seen the <|apothecary|> 's wares ?
```

## Training Data Sources

The tokenizer was trained on 108M characters from 94 historical sources including:

- **London Lives 1690-1800** - 240,000 manuscript pages
- **Old Bailey Online** - 197,000+ trial accounts
- **Project Gutenberg** - 60+ historical texts
- **Samuel Pepys' Diary** - Personal accounts of London life
- **John Stow's Survey of London** - 16th-century city description
- **Parish Records** - Baptisms, marriages, burials
- **London Gazette** - Official government newspaper

## Performance Metrics

- **Training Data**: 108,368,442 characters
- **Text Sources**: 94 files
- **Vocabulary Size**: 30,000 tokens
- **Special Tokens**: 150+ historical and London-specific
- **Training Time**: ~3-5 minutes on modern hardware
- **Model Size**: ~200MB (tokenizer files)

## Future Enhancements

### Potential Additions
- **Dialect Tokens**: Regional variations within London
- **Occupational Slang**: Trade-specific terminology
- **Social Register**: Formal vs. informal speech patterns
- **Temporal Markers**: More specific time references

### Optimization Opportunities
- **Frequency Analysis**: Remove low-frequency special tokens
- **Context Clustering**: Group related terms
- **Dynamic Vocabulary**: Adapt based on specific text types

## 🔧 **Troubleshooting Tokenization Issues**

### **Common Problems and Solutions**

**Problem: Word not tokenizing as expected**
```bash
# Check if word should be a special token
grep -i "your_word" TOKENIZER_VOCABULARY.md

# Test tokenization directly
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('path/to/tokenizer')
tokens = tokenizer.encode('your_text_here')
print(tokens)
print(tokenizer.decode(tokens))
"
```

**Problem: Inefficient tokenization (too many subwords)**
- Check if the word should be added as a special token
- Review frequency in your dataset - high-frequency words should be single tokens
- Consider if the word is London-specific or historically important

**Problem: Special tokens not working**
```bash
# Verify tokenizer has special tokens
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('path/to/tokenizer')
print('Special tokens:', tokenizer.special_tokens_map)
print('Vocab size:', tokenizer.vocab_size)
"
```

### **Integration with Token Counting**

When debugging tokenization efficiency:
1. **Use TOKEN_COUNTING_GUIDE.md** to measure actual token counts
2. **Use TOKENIZER_VOCABULARY.md** to understand why certain words tokenize the way they do
3. **Compare expected vs actual** tokenization efficiency

**Example Debugging Workflow:**
```bash
# 1. Count tokens in your data
python 02_data_collection/count_tokens.py --data_dir your_data

# 2. Check if problematic words are special tokens
grep -i "problematic_word" TOKENIZER_VOCABULARY.md

# 3. Test specific tokenization
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('path/to/tokenizer')
text = 'Your problematic text here'
tokens = tokenizer.encode(text)
print(f'Text: {text}')
print(f'Tokens: {tokens}')
print(f'Decoded: {tokenizer.decode(tokens)}')
"
```

---

