# London Historical LLM - Inference Guide

**✅ Unified inference system for both SLM (117M) and Regular (354M) models - FULLY WORKING**

> **Status**: Both PyTorch checkpoint inference and Hugging Face model inference are working perfectly!

## ✅ **Current Status**

| **Inference Type** | **SLM (117M)** | **Regular (354M)** | **Status** |
|-------------------|----------------|-------------------|------------|
| **PyTorch Checkpoints** | ✅ Working | ✅ Working | Both models load and generate correctly |
| **Hugging Face Models** | ✅ Published | 🔄 Ready to publish | SLM published, Regular ready |
| **Local Testing** | ✅ Tested | ✅ Tested | Both models tested on remote Ubuntu |

## 🚀 **Quick Start**

### **1. Published Models (Recommended)**
```bash
# Test published SLM model
python inference_unified.py --published --model_type slm

# Test published Regular model (when available)
python inference_unified.py --published --model_type regular

# Interactive mode with published SLM
python inference_unified.py --published --interactive
```

### **2. Local Models**
```bash
# Load local model directory
python inference_unified.py --model_path ./09_models/checkpoints/checkpoint-1000

# Load with custom tokenizer
python inference_unified.py --model_path ./09_models/checkpoints/checkpoint-1000 --tokenizer_path ./09_models/tokenizers/london_historical_tokenizer
```

### **3. PyTorch Checkpoints (During Training)**
```bash
# Test SLM PyTorch checkpoint (117M parameters)
python 06_inference/inference_pytorch.py \
  --checkpoint 09_models/checkpoints/slm/checkpoint-4000.pt \
  --prompt "In the year 1834, I walked through the streets of London and witnessed"

# Test Regular PyTorch checkpoint (354M parameters)
python 06_inference/inference_pytorch.py \
  --checkpoint 09_models/checkpoints/checkpoint-60001.pt \
  --prompt "In the year 1834, I walked through the streets of London and witnessed"

# Interactive mode with checkpoint
python 06_inference/inference_pytorch.py --checkpoint ./09_models/checkpoints/checkpoint-25500.pt --interactive
```

### **4. Test Published Models**
```bash
# Test all published models
python test_published_models.py

# Test only SLM
python test_published_models.py --model_type slm

# Test custom model
python test_published_models.py --model_name "your-model-name"
```

## 📋 **Script Overview**

### **`inference_unified.py`** - 🎯 **Main Script**
**Purpose**: Unified interface for all model types and formats
**Supports**: 
- Published models from Hugging Face Hub
- Local Hugging Face format models
- Both SLM and Regular models
- Auto-detection of model type

**Key Features**:
- Auto-detects model type (SLM vs Regular)
- Optimized parameters for each model type
- Interactive and demo modes
- Single prompt generation

### **`inference_pytorch.py`** - 🔧 **PyTorch Checkpoints**
**Purpose**: Direct inference from PyTorch `.pt` checkpoint files
**Use Case**: Testing during training before HF conversion
**Supports**:
- Raw PyTorch checkpoints
- Both SLM and Regular models
- Custom model architectures

### **`test_published_models.py`** - 🧪 **Model Testing**
**Purpose**: Comprehensive testing of published models
**Features**:
- Tests both SLM and Regular models
- Performance benchmarking
- Error reporting
- Load time and generation speed metrics

## 📊 **Expected Output Examples**

### **SLM Model Output (117M parameters)**
```bash
python 06_inference/inference_pytorch.py \
  --checkpoint 09_models/checkpoints/slm/checkpoint-4000.pt \
  --prompt "In the year 1834, I walked through the streets of London and witnessed"

# Expected output:
🏛️ London Historical LLM - PyTorch Checkpoint Inference
======================================================================
2025-09-22 14:41:33,061 - INFO - 📂 Loading PyTorch checkpoint: 09_models/checkpoints/slm/checkpoint-4000.pt
2025-09-22 14:41:33,061 - INFO - 🎯 Model type: slm - Small Language Model (117M parameters)
2025-09-22 14:41:33,061 - INFO - 🔤 Loading tokenizer from: /home/amit/helloLondon/09_models/tokenizers/london_historical_tokenizer
2025-09-22 14:41:33,113 - INFO - 🤖 Loading model from PyTorch checkpoint...
2025-09-22 14:41:33,820 - INFO - Using 'model' from checkpoint
2025-09-22 14:41:33,821 - INFO - 🤖 Attempting to load as Hugging Face model...
2025-09-22 14:41:33,821 - INFO - 🔄 Attempting to load as raw PyTorch checkpoint...
2025-09-22 14:41:33,821 - INFO - 🤖 Using SimpleGPT architecture for SLM...
2025-09-22 14:41:35,828 - INFO - 🔧 Detected torch.compile checkpoint, stripping _orig_mod. prefixes...
2025-09-22 14:41:35,829 - INFO - ✅ Cleaned 76 parameters
2025-09-22 14:41:35,904 - INFO - ✅ Loaded as raw PyTorch checkpoint
2025-09-22 14:41:35,904 - INFO - ✅ Model loaded successfully!

📊 Model Information:
   Type: SLM
   Description: Small Language Model (117M parameters)
   Device: cuda
   Vocabulary size: 30,000
   Model parameters: ~80,449,536

📝 Prompt: In the year 1834, I walked through the streets of London and witnessed
🤖 Generating...

📖 Generated text:
--------------------------------------------------
##cy . The old manor is of more than a century later , and is in the same house . It is not far from the earliest times that I was a son of the Earl of Norfolk . I have been told that when I was Lord of Leicester he came to London , I became a Lord Mayor , and I became a Lord Mayor , and he died in the middle of the year . I took this title from the Earl of Essex to a Lord Mayor , and I became my Lord Mayor . I had the honour of his Majesty , and I had a mind to see the Duke of Kent . I was called to the Duke of Kent . I then wrote to the Lord Mayor , and to the Lords Justices of the Treasury , to whom I had a share in the power of the Lord Mayor . My Lord Mayor , I gave him a cup of wine , to his great astonishment . I told him that he was a master of the City , and a great number of the King ' s company were there . I was informed that he had a great many of the poor old inhabitants of the city . I had no occasion to say that I had to learn from the year the Lord Mayor had taken his seat in the street . But I was told that it was not an old house where I had any share in the City , and that I had a share in the City , and that the city was an old church , called the City , and a church where I had a house of some importance . I had a lease of the old parish of St . Paul , and the old parish of St . Paul , which I had built , I bought it , and so I went home , and made a great number of the old church - house . I went to the church and saw it . I had a great many chambers and a library , and a small room , with a table in it . I took it to the church , and set it down upon my table . I had to write , and found it in the hands of the Lord Mayor . I took it out , and went out to the church . I told him that I would not go any longer , and I went home and sent it to the church . I had a bed , and slept soundly . I went to my house , and took it and slept . I went to the church , and there sat with my wife and I alone , and so went to the church , and there I had a chamber in my chamber , and a chamber in the garden ; and I slept . I went out of the church , and there walked to the church . I was called by the Lord Mayor , and he was there . I went home to the church , where I found my Lord Mayor and the Lord Mayor , and I went to the church to hear how it had been in the church , which I was afeard . I went home and found him there , and he took me to the church , and there I found him in the church , and in a little garden , and there saw a very old church , and a small door behind it . After a little stay I saw a woman and a woman who had a mind to walk and see me , and so went home and to supper , and then to bed . 14th . Up , and to the church , where I found my Lord Mayor . I walked to the church , and I left it , and walked with me to the church . Then to the church again . Then home with my wife , who had a long life to tell me that she did make a service for me , and she did give me a very kind of clothes , and then went out to the church and found me . So I walked to the church , where I found the old church called the " Trinity House . " Here I was and the church and garden of St . James , and there sat a long time . I went out and looked into the church , and saw the old church in the church there . I went to the church , and there came to look over the old church and the church and the church there , and there stood the old church , but it was not well , and the church of the church was full of doors , and a well - built and perfect stone and a high cross - stone . Then to the church again . Then to the church and the church there was a very old church , and a very old church in the church . Then to the church and church again . I went to a church , where a great church and an old church ( very handsome ) , and in the church and a high cross . I took it up , and walked to the church , and there I found a great chapel with the great bell , and a very large stone cross , and a long cross in the church . Then to the church again , and there stood the church and a low church with a low church and a large church . Then to the church and there stood a church with a large church and a long stone . Then to
--------------------------------------------------

✅ Inference completed!
```

### **Regular Model Output (354M parameters)**
```bash
python 06_inference/inference_pytorch.py \
  --checkpoint 09_models/checkpoints/checkpoint-60001.pt \
  --prompt "In the year 1834, I walked through the streets of London and witnessed"

# Expected output:
🏛️ London Historical LLM - PyTorch Checkpoint Inference
======================================================================
2025-09-22 14:40:26,424 - INFO - 📂 Loading PyTorch checkpoint: 09_models/checkpoints/checkpoint-60001.pt
2025-09-22 14:40:26,425 - INFO - 🎯 Model type: regular - Regular Language Model (354M parameters)
2025-09-22 14:40:26,425 - INFO - 🔤 Loading tokenizer from: /home/amit/helloLondon/09_models/tokenizers/london_historical_tokenizer
2025-09-22 14:40:26,477 - INFO - 🤖 Loading model from PyTorch checkpoint...
2025-09-22 14:40:27,526 - INFO - Using 'model' from checkpoint
2025-09-22 14:40:27,526 - INFO - 🤖 Attempting to load as Hugging Face model...
2025-09-22 14:40:27,527 - INFO - 🔄 Attempting to load as raw PyTorch checkpoint...
2025-09-22 14:40:27,527 - INFO - 🤖 Using GPT architecture for regular model...
2025-09-22 14:40:31,615 - INFO - 🔧 Detected torch.compile checkpoint, stripping _orig_mod. prefixes...
2025-09-22 14:40:31,615 - INFO - ✅ Cleaned 148 parameters
2025-09-22 14:40:31,878 - INFO - ✅ Loaded as raw PyTorch checkpoint
2025-09-22 14:40:31,878 - INFO - ✅ Model loaded successfully!

📊 Model Information:
   Type: REGULAR
   Description: Regular Language Model (354M parameters)
   Device: cuda
   Vocabulary size: 30,000
   Model parameters: ~233,095,168

📝 Prompt: In the year 1834, I walked through the streets of London and witnessed
🤖 Generating...

📖 Generated text:
--------------------------------------------------
##urs , and saw the people assembled there , in all the show of life , with their children , playing and playing at cards . My fellow - travellers , I found , had been playing at cards , and had heard , from time to time , the talk of the country people . There were many of them , I was told , talking about the country people , and going out in the fields and the races of the country people , and in the course of the year , and of the races of men . I saw , as I passed , that there were few people in the town who had been in the town , and that there were some , and there were some who had been there once . I was told that there were many who had been there once , and that there were at least fifty who had been there once . I heard of one man , who was a great while walking along the streets , talking about the country , and being then in a great passion with the people about him , and being very angry with him , I went up to my lodging , and told him that I was very sorry he had not been there . He told me that he was going into the country too , but that he was very glad to see me , and that he was afraid to hear of me , and that I was afraid to speak to him , for he was sure I should be so , though he did not like to speak to me . I asked him what he thought of the country . He told me , that though he did not know himself to be a stranger , yet he was very much pleased with me , and that he would be glad to see me , and that he would take me in ; and that if he was not a stranger , he would be glad to see me , and I would try to make him happy . I told him that I was very glad to see him , and that if I would serve him , and if I would let him go , he should be very glad to see me , and that perhaps I might be glad to see him again . I told him , also , that he was glad to see me , and that he could not be so kind to me as I could be to him , and that if he would take me in , he would give me leave to take care of me , and that I would do him a kindness and a kindness that would help me , which he would do very willingly , and that I would do him good . He told me also that he had heard from my brother that he had heard from me of his father ' s being dead , and that he had heard of him , and that he had been very kind to me , and that he had been very kind to me and had told me he was very kind to me , and that he was to me , though he knew very well what to say , yet he had not told me that he knew of my being in the country , and that he had been told that I was indeed the author of all that he knew of him , and that he had a great deal of respect for me , and that he had not been told that I was not the author of all that I had heard , but that he had never heard that I was in the town , or that he had been there before or since . I told him that I had heard of him at his house , and that he was in such a humour that he could not see me , but he would tell me that he was not very kind to me , and that he would be glad to see me again , and that perhaps he would tell me that if I would not be pleased to see him again , I should be glad to see him again ; and that if I would promise to do him the favour to tell me that I would not be pleased to see him again if I would be pleased to see him again , and that I would do him a kindness , and that I would do him a kindness , which I did not like , and that indeed he would be glad to hear me talk so kindly . He told me that he had a very good opinion of me and my brother , and that he would come and stay a week or two longer with me , and that he would be glad to see me , and that he should not be able to do anything that was possible for me to do , for indeed he would not let me come to him at all , but was very kind to me in that manner ; and that , in short , I hoped he would do me the honour to tell me that he thought it would be as well , and that if I would do him a kindness , and if he would do me the favour to let me go . This was a great while before I could bear it , but it was not so , and so it was , and so I let him go by coach to my brother ' s , and we to my brother ' s , and there I did
--------------------------------------------------

✅ Inference completed!
```

## 🎯 **Model Types**

### **SLM (Small Language Model)**
- **Parameters**: ~117M
- **Published**: `bahree/london-historical-slm`
- **Max Length**: 512 tokens
- **Use Case**: Fast inference, resource-constrained environments
- **Temperature**: 0.8 (more creative)
- **Top-p**: 0.9, Top-k: 50

### **Regular (Language Model)**
- **Parameters**: ~354M
- **Published**: `bahree/london-historical-llm` (when available)
- **Max Length**: 1024 tokens
- **Use Case**: Higher quality, more detailed responses
- **Temperature**: 0.7 (more focused)
- **Top-p**: 0.95, Top-k: 40

## **Historical Prompt Library**

### **Tudor Period (1500-1600) - Renaissance & Reformation**

#### **Diary Entries**
```
"Today I walked through the streets of London and witnessed..."
"On this day in 1558, Queen Mary has died and..."
"The plague has taken hold of the city, and I fear..."
"In the year of our Lord 1520, I have seen..."
"The new learning from the universities has reached London..."
```

#### **Court Records**
```
"The case of John Smith versus the Corporation of London regarding..."
"In the matter of the theft of goods from the market at..."
"The defendant stands accused of heresy against the Church of England..."
"The plaintiff seeks redress for damages to his property..."
"The court finds the accused guilty of..."
```

#### **Religious Documents**
```
"The Pope has excommunicated King Henry VIII, and..."
"The new Book of Common Prayer has been introduced..."
"The monasteries are being dissolved, and their wealth..."
"The Protestant faith is spreading through the city..."
```

### **🔥 Stuart Period (1600-1700) - Civil War, Plague & Fire**

#### **Newspaper Articles**
```
"The Great Fire of London has consumed much of the city, and..."
"King Charles I has been executed, and the Commonwealth begins..."
"The plague has returned to London, with over 100,000 dead..."
"Parliament has declared war on the King, and..."
"The city is in chaos as the Roundheads and Cavaliers..."
```

#### **Political Pamphlets**
```
"The people of London demand their rights and freedoms..."
"Parliament must act to protect the citizens from..."
"The monarchy has failed the people of England..."
"The Levellers call for equality and justice..."
"The Diggers seek to reclaim the common land..."
```

#### **Personal Letters**
```
"My dearest friend, I write to you from the besieged city..."
"The King's forces have been defeated, and..."
"The plague has taken my beloved wife, and I..."
"The fire has destroyed our home, but we shall rebuild..."
```

### **🏭 Georgian Period (1700-1800) - Enlightenment & Industrial Revolution**

#### **Merchant Accounts**
```
"The trade in spices from the East Indies has brought great wealth..."
"Our ships have returned from the Americas with..."
"The cost of living in London has risen dramatically due to..."
"The new factories along the Thames are producing..."
"The East India Company has established a monopoly on..."
```

#### **Travel Journals**
```
"As I journeyed through the streets of London, I observed..."
"The new buildings and improvements to the city are remarkable..."
"The Thames flows through the heart of the city, carrying..."
"The coffee houses are filled with learned men discussing..."
"The new bridges across the river have improved..."
```

#### **Scientific Treatises**
```
"The principles of natural philosophy can be observed in..."
"The new steam engines are revolutionizing industry..."
"The study of anatomy has revealed the secrets of..."
"The Royal Society has published new findings on..."
```

### **👑 Regency/Victorian Early (1800-1850) - Social Reform**

#### **Letters**
```
"My dearest friend, I write to you from the bustling streets of London..."
"The conditions in the workhouses are deplorable, and something must be done..."
"The new railway has connected London to the rest of England..."
"The Chartists are demanding political reform, and..."
"The factory workers are organizing for better conditions..."
```

#### **Sermons**
```
"Brothers and sisters, we gather today to discuss the moral decay..."
"The poor and destitute of London cry out for our help..."
"God has blessed this great city with prosperity, but we must not forget..."
"The new Poor Law has failed to address the suffering..."
"The temperance movement calls for moderation in all things..."
```

#### **Social Commentary**
```
"The gap between rich and poor in London has never been greater..."
"The new middle class is emerging, and with it..."
"The slums of the East End are a blight on our city..."
"The new public health measures are insufficient to..."
```

## 🎭 **Creative & Literary Prompts**

### **Play Scripts**
```
"Enter HAMLET, soliloquizing about the state of London..."
"The scene opens in a tavern on Fleet Street, where..."
"LADY MACBETH: 'Out, out, brief candle! The streets of London...'"
"Enter FALSTAFF, drunk and merry, speaking of..."
"The curtain rises on a scene of London in the year..."
```

### **Poetry**
```
"O London, city of my heart, where..."
"The Thames flows on, carrying the dreams of..."
"In the shadow of St. Paul's, I find..."
"The fog rolls in from the river, and..."
"The bells of Westminster chime the hour..."
```

### **Novels**
```
"Chapter One: In which our hero arrives in London..."
"The fog was thick on the morning of..."
"Elizabeth Bennet walked through the streets of London..."
"The mysterious stranger appeared at the door of..."
```

## 🏥 **Medical & Scientific Prompts**

### **Medical Treatises**
```
"The treatment of the ague requires careful attention to..."
"The miasmic air of the city contributes to the spread of..."
"Bloodletting, when performed correctly, can alleviate..."
"The new understanding of anatomy has revealed..."
"The use of mercury in treating syphilis is..."
```

### **Recipe Books**
```
"To make a proper English pudding, one must first..."
"The art of brewing ale in London requires..."
"For a remedy against the plague, take..."
"The preparation of medicinal herbs requires..."
"To preserve meat for the winter months..."
```

### **Scientific Observations**
```
"The microscope has revealed the hidden world of..."
"The new theories of gravity explain the motion of..."
"The study of electricity has opened new possibilities..."
"The classification of plants and animals has..."
```

## 🏛️ **London-Specific Prompts**

### **Famous Locations**
```
"St. Paul's Cathedral stands as a testament to..."
"The Tower of London has witnessed many changes..."
"Fleet Street is alive with the sound of..."
"Westminster Abbey has been the site of..."
"The Thames flows through the heart of the city..."
```

### **Historical Events**
```
"The Great Fire of 1666 began in Pudding Lane..."
"The Plague of 1665 claimed over 100,000 lives..."
"The Gunpowder Plot of 1605 was discovered..."
"The execution of Charles I took place at..."
"The coronation of Queen Victoria was celebrated..."
```

## 🧪 **Testing Strategies**

### **1. Period Accuracy Testing**
- Test prompts from different historical periods
- Verify period-appropriate language and vocabulary
- Check for anachronistic references

### **2. Document Type Testing**
- Test different document formats (diary, letter, court record, etc.)
- Verify appropriate formatting and structure
- Check for consistent voice and tone

### **3. Knowledge Testing**
- Ask about specific historical events
- Test knowledge of London geography
- Verify understanding of social conditions

### **4. Creative Testing**
- Test creative writing abilities
- Check for consistent character voice
- Verify narrative coherence

### **5. Technical Testing**
- Test with very short prompts
- Test with very long prompts
- Test with incomplete sentences
- Test with non-English words

## 📊 **Evaluation Criteria**

### **Historical Accuracy (1-10)**
- Period-appropriate language
- Accurate historical references
- Consistent with historical context

### **Writing Quality (1-10)**
- Grammar and syntax
- Vocabulary richness
- Sentence structure

### **Creativity (1-10)**
- Originality of content
- Narrative coherence
- Character development

### **London Knowledge (1-10)**
- Geographic accuracy
- Historical event knowledge
- Social context understanding

## 🎯 **Sample Testing Workflow**

### **Phase 1: Basic Functionality**
1. Test simple prompts
2. Verify model loads correctly
3. Check output format

### **Phase 2: Historical Accuracy**
1. Test Tudor period prompts
2. Test Stuart period prompts
3. Test Georgian period prompts
4. Test Victorian period prompts

### **Phase 3: Document Types**
1. Test diary entries
2. Test letters
3. Test court records
4. Test newspaper articles

### **Phase 4: Creative Writing**
1. Test play scripts
2. Test poetry
3. Test novels
4. Test sermons

### **Phase 5: Edge Cases**
1. Test very short prompts
2. Test very long prompts
3. Test incomplete sentences
4. Test non-English words

## 🔧 **Troubleshooting**

### **Common Issues**
- **Model not loading**: Check model path and configuration
- **Poor output quality**: Try different prompts or adjust parameters
- **Inconsistent responses**: Test with same prompt multiple times
- **Historical inaccuracies**: Verify training data quality

### **Performance Tips**
- Use shorter prompts for faster responses
- Adjust temperature for creativity vs. accuracy
- Use top_p for better word selection
- Adjust max_length for response length

## 📚 **Additional Resources**

### **Historical Context**
- [Tudor England (1500-1600)](https://en.wikipedia.org/wiki/Tudor_period)
- [Stuart England (1600-1700)](https://en.wikipedia.org/wiki/Stuart_period)
- [Georgian England (1700-1800)](https://en.wikipedia.org/wiki/Georgian_era)
- [Victorian England (1800-1900)](https://en.wikipedia.org/wiki/Victorian_era)

### **London History**
- [History of London](https://en.wikipedia.org/wiki/History_of_London)
- [Great Fire of London](https://en.wikipedia.org/wiki/Great_Fire_of_London)
- [London Plague](https://en.wikipedia.org/wiki/Great_Plague_of_London)

### **Literary References**
- [Shakespeare's London](https://en.wikipedia.org/wiki/Shakespeare%27s_London)
- [Dickens's London](https://en.wikipedia.org/wiki/Charles_Dickens)
- [Victorian Literature](https://en.wikipedia.org/wiki/Victorian_literature)

## 🎉 **Getting Started**

1. **Start with simple prompts** to test basic functionality
2. **Try different historical periods** to test knowledge
3. **Test various document types** to test versatility
4. **Use creative prompts** to test imagination
5. **Evaluate output quality** using the criteria above

**Happy testing!** 🚀✨
