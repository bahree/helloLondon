#!/usr/bin/env python3
"""
Synthetic Historical Data Generator
Generates additional historical London data using local or cloud LLMs
"""

import os
import sys
import json
import argparse
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import global configuration
from config import config

class SyntheticDataGenerator:
    """Generate synthetic historical London data using LLMs"""
    
    def __init__(self, output_dir: Path = None, model_type: str = "ollama"):
        self.output_dir = output_dir or config.london_historical_data / "synthetic"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_type = model_type  # "ollama", "openai", "anthropic"
        self.model_name = self.get_model_name()
        
        # Setup logging
        self.setup_logging()
        
        # Historical periods and contexts
        self.historical_periods = [
            {"start": 1500, "end": 1600, "name": "Tudor Period", "context": "Renaissance, religious reformation, early modern London"},
            {"start": 1600, "end": 1700, "name": "Stuart Period", "context": "Civil war, restoration, scientific revolution, plague and fire"},
            {"start": 1700, "end": 1800, "name": "Georgian Period", "context": "Enlightenment, industrial revolution, empire expansion"},
            {"start": 1800, "end": 1850, "name": "Regency/Victorian Early", "context": "Industrial revolution, social reform, urbanization"}
        ]
        
        # London locations and contexts
        self.london_locations = [
            "Cheapside", "Fleet Street", "Ludgate Hill", "Covent Garden", "Westminster",
            "Southwark", "Whitechapel", "Spitalfields", "Clerkenwell", "Holborn",
            "Tower Hill", "Billingsgate", "Smithfield", "Newgate", "Old Bailey",
            "St. Paul's Cathedral", "Westminster Abbey", "London Bridge", "Thames Street"
        ]
        
        # Historical document types
        self.document_types = [
            "diary_entry", "letter", "newspaper_article", "court_record", "parish_register",
            "merchant_account", "guild_record", "travel_journal", "political_pamphlet",
            "sermon", "play_script", "poem", "recipe_book", "medical_treatise"
        ]
        
        # Statistics
        self.stats = {
            'documents_generated': 0,
            'total_words': 0,
            'periods_covered': set(),
            'document_types_used': set(),
            'start_time': datetime.now()
        }
    
    def get_model_name(self) -> str:
        """Get the appropriate model name based on type"""
        if self.model_type == "ollama":
            return "phi3:latest"  # or "llama3.1:latest", "mistral:latest"
        elif self.model_type == "openai":
            return "gpt-4"
        elif self.model_type == "anthropic":
            return "claude-3-sonnet-20240229"
        else:
            return "phi3:latest"
    
    def setup_logging(self):
        """Setup logging for the generator"""
        log_file = self.output_dir / "synthetic_generation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_prompts(self, num_documents: int = 100) -> List[Dict[str, Any]]:
        """Generate diverse prompts for historical document creation"""
        prompts = []
        
        for i in range(num_documents):
            period = random.choice(self.historical_periods)
            location = random.choice(self.london_locations)
            doc_type = random.choice(self.document_types)
            
            # Generate specific year within period
            year = random.randint(period["start"], period["end"])
            
            # Create context-specific prompts
            if doc_type == "diary_entry":
                prompt = f"""Write a detailed diary entry from {year} by a {random.choice(['merchant', 'noble', 'artisan', 'clergyman', 'scholar'])} living in {location}, London. 

Context: {period['context']}
Location: {location}
Year: {year}

Include:
- Daily activities and observations
- Social interactions and events
- Economic concerns and trade
- Weather and seasonal changes
- Personal reflections on London life
- Historical events of the time

Write in authentic period language and style. Make it 300-500 words."""
            
            elif doc_type == "letter":
                prompt = f"""Write a letter from {year} sent from {location}, London to a family member or business associate.

Context: {period['context']}
Location: {location}
Year: {year}

Include:
- News about London events and changes
- Personal or business matters
- Observations about the city and its people
- References to specific locations and landmarks
- Period-appropriate language and concerns

Write in authentic period language. Make it 200-400 words."""
            
            elif doc_type == "newspaper_article":
                prompt = f"""Write a newspaper article from {year} about events in {location}, London.

Context: {period['context']}
Location: {location}
Year: {year}

Include:
- Local news and events
- Social and economic developments
- Political or religious matters
- Crime, accidents, or notable incidents
- Weather and seasonal reports
- Advertisements or announcements

Write in period-appropriate journalistic style. Make it 250-400 words."""
            
            elif doc_type == "court_record":
                prompt = f"""Write a court record from {year} of a case heard in {location}, London.

Context: {period['context']}
Location: {location}
Year: {year}

Include:
- Case details and charges
- Witness testimonies
- Evidence presented
- Verdict and sentence
- Social context and implications
- Period-appropriate legal language

Write in formal legal style of the period. Make it 300-500 words."""
            
            elif doc_type == "merchant_account":
                prompt = f"""Write a merchant's account book entry from {year} for business in {location}, London.

Context: {period['context']}
Location: {location}
Year: {year}

Include:
- Detailed financial transactions
- Inventory and goods
- Trade partners and locations
- Prices and currencies
- Business challenges and opportunities
- Economic conditions

Write in period-appropriate business style. Make it 200-400 words."""
            
            else:
                # Generic historical document
                prompt = f"""Write a {doc_type} from {year} related to {location}, London.

Context: {period['context']}
Location: {location}
Year: {year}
Document Type: {doc_type}

Create an authentic historical document that would have been written during this period. Include specific details about London life, social conditions, and historical context. Use period-appropriate language and style. Make it 250-400 words."""
            
            prompts.append({
                'prompt': prompt,
                'metadata': {
                    'year': year,
                    'period': period['name'],
                    'location': location,
                    'document_type': doc_type,
                    'context': period['context']
                }
            })
        
        return prompts
    
    async def generate_with_ollama(self, prompt: str) -> str:
        """Generate text using Ollama API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        self.logger.error(f"Ollama API error: {response.status}")
                        return ""
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            return ""
    
    async def generate_with_openai(self, prompt: str) -> str:
        """Generate text using OpenAI API"""
        try:
            import openai
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a historian specializing in early modern London. Write authentic historical documents with period-appropriate language and accurate historical context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return ""
    
    async def generate_document(self, prompt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single historical document"""
        prompt = prompt_data['prompt']
        metadata = prompt_data['metadata']
        
        self.logger.info(f"Generating {metadata['document_type']} from {metadata['year']}...")
        
        # Generate text based on model type
        if self.model_type == "ollama":
            text = await self.generate_with_ollama(prompt)
        elif self.model_type == "openai":
            text = await self.generate_with_openai(prompt)
        else:
            self.logger.error(f"Unsupported model type: {self.model_type}")
            return None
        
        if not text.strip():
            self.logger.warning(f"Empty response for {metadata['document_type']} from {metadata['year']}")
            return None
        
        # Create document
        document = {
            'text': text.strip(),
            'metadata': metadata,
            'generation_info': {
                'model': self.model_name,
                'model_type': self.model_type,
                'generated_at': datetime.now().isoformat(),
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        }
        
        # Update statistics
        self.stats['documents_generated'] += 1
        self.stats['total_words'] += document['generation_info']['word_count']
        self.stats['periods_covered'].add(metadata['period'])
        self.stats['document_types_used'].add(metadata['document_type'])
        
        return document
    
    async def generate_batch(self, prompts: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """Generate documents in batches to avoid overwhelming the API"""
        documents = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            tasks = [self.generate_document(prompt_data) for prompt_data in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            for result in batch_results:
                if isinstance(result, dict):
                    documents.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error in batch processing: {result}")
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        return documents
    
    def save_documents(self, documents: List[Dict[str, Any]]):
        """Save generated documents to files"""
        self.logger.info(f"Saving {len(documents)} documents...")
        
        # Save individual documents
        for i, doc in enumerate(documents):
            filename = f"synthetic_{doc['metadata']['year']}_{doc['metadata']['document_type']}_{i:03d}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Generated: {doc['generation_info']['generated_at']}\n")
                f.write(f"Model: {doc['generation_info']['model']}\n")
                f.write(f"Year: {doc['metadata']['year']}\n")
                f.write(f"Period: {doc['metadata']['period']}\n")
                f.write(f"Location: {doc['metadata']['location']}\n")
                f.write(f"Type: {doc['metadata']['document_type']}\n")
                f.write("=" * 80 + "\n\n")
                f.write(doc['text'])
        
        # Save metadata
        metadata_file = self.output_dir / "synthetic_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generation_info': {
                    'model_type': self.model_type,
                    'model_name': self.model_name,
                    'total_documents': len(documents),
                    'total_words': sum(doc['generation_info']['word_count'] for doc in documents),
                    'periods_covered': list(self.stats['periods_covered']),
                    'document_types': list(self.stats['document_types_used']),
                    'generation_time': (datetime.now() - self.stats['start_time']).total_seconds()
                },
                'documents': documents
            }, f, indent=2)
        
        # Create corpus file
        corpus_file = self.output_dir / "synthetic_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            f.write("Synthetic Historical London Corpus\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Documents: {len(documents)}\n")
            f.write(f"Total Words: {sum(doc['generation_info']['word_count'] for doc in documents):,}\n")
            f.write("=" * 50 + "\n\n")
            
            for doc in documents:
                f.write(f"\n--- {doc['metadata']['document_type'].title()} from {doc['metadata']['year']} ---\n")
                f.write(doc['text'])
                f.write("\n" + "=" * 80 + "\n")
        
        self.logger.info(f"✅ Documents saved to {self.output_dir}")
        self.logger.info(f"📚 Corpus file: {corpus_file}")
    
    def print_summary(self):
        """Print generation summary"""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        print("\n" + "=" * 70)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("=" * 70)
        print(f"Model: {self.model_name} ({self.model_type})")
        print(f"Documents generated: {self.stats['documents_generated']}")
        print(f"Total words: {self.stats['total_words']:,}")
        print(f"Average words per document: {self.stats['total_words'] // max(1, self.stats['documents_generated'])}")
        print(f"Generation time: {duration:.1f} seconds")
        print(f"Periods covered: {', '.join(sorted(self.stats['periods_covered']))}")
        print(f"Document types: {', '.join(sorted(self.stats['document_types_used']))}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
    
    async def run_generation(self, num_documents: int = 100, batch_size: int = 5):
        """Run the complete synthetic data generation process"""
        self.logger.info(f"🚀 Starting synthetic data generation with {self.model_name}")
        self.logger.info(f"Target: {num_documents} documents")
        
        # Generate prompts
        prompts = self.generate_prompts(num_documents)
        self.logger.info(f"Generated {len(prompts)} prompts")
        
        # Generate documents
        documents = await self.generate_batch(prompts, batch_size)
        
        if not documents:
            self.logger.error("No documents generated!")
            return False
        
        # Save documents
        self.save_documents(documents)
        
        # Print summary
        self.print_summary()
        
        return True

async def main():
    """Main function for synthetic data generation"""
    parser = argparse.ArgumentParser(description="Generate synthetic historical London data")
    parser.add_argument("--num_documents", type=int, default=100,
                       help="Number of documents to generate")
    parser.add_argument("--model_type", choices=["ollama", "openai", "anthropic"], default="ollama",
                       help="Model type to use for generation")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="Batch size for concurrent generation")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (uses global config if not specified)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model_type=args.model_type
    )
    
    # Check if Ollama is running (for local models)
    if args.model_type == "ollama":
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status != 200:
                        print("❌ Ollama is not running. Start it with: ollama serve")
                        return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("Start Ollama with: ollama serve")
            return False
    
    # Run generation
    success = await generator.run_generation(
        num_documents=args.num_documents,
        batch_size=args.batch_size
    )
    
    if success:
        print("✅ Synthetic data generation completed successfully!")
    else:
        print("❌ Synthetic data generation failed")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
