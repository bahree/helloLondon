#!/usr/bin/env python3
"""
Historical Evaluation Dataset for London Historical SLM (1500-1850)
Creates specialized test cases for historical accuracy, language patterns, and period-specific knowledge
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class HistoricalEvaluationDataset:
    """Dataset for evaluating historical London SLM"""
    
    def __init__(self):
        self.test_cases = []
        self._create_historical_test_cases()
    
    def _create_historical_test_cases(self):
        """Create comprehensive test cases for historical evaluation"""
        
        # 1. Historical Accuracy Tests
        historical_accuracy_tests = [
            {
                'category': 'plague_1665',
                'prompt': 'In 1665, London was struck by',
                'expected_keywords': ['plague', 'pestilence', 'disease', 'death', 'bubonic'],
                'expected_phrases': ['Great Plague', 'bubonic plague', 'Black Death'],
                'context': 'The Great Plague of London was a bubonic plague epidemic that occurred in 1665-1666.',
                'expected_facts': ['plague', '1665', 'London', 'bubonic', 'epidemic', 'Black Death']
            },
            {
                'category': 'fire_1666',
                'prompt': 'In 1666, London experienced',
                'expected_keywords': ['fire', 'burning', 'destruction', 'flames', 'conflagration'],
                'expected_phrases': ['Great Fire', 'London Fire', 'Great Fire of London'],
                'context': 'The Great Fire of London was a major conflagration that swept through London in 1666.',
                'expected_facts': ['fire', '1666', 'London', 'Great Fire', 'conflagration']
            },
            {
                'category': 'royalty_charles_ii',
                'prompt': 'The King of England in 1665 was',
                'expected_keywords': ['Charles', 'King', 'monarch', 'throne', 'Charles II'],
                'expected_phrases': ['Charles II', 'King Charles', 'King Charles II'],
                'context': 'Charles II was King of England, Scotland, and Ireland from 1660 to 1685.',
                'expected_facts': ['Charles II', 'King', '1660', '1685', 'England']
            },
            {
                'category': 'religion_anglican',
                'prompt': 'The Church of England was',
                'expected_keywords': ['Anglican', 'Protestant', 'Church', 'religion', 'established'],
                'expected_phrases': ['established church', 'Anglican Church', 'Church of England'],
                'context': 'The Church of England is the established Christian church in England.',
                'expected_facts': ['Anglican', 'Protestant', 'established', 'Christian', 'England']
            },
            {
                'category': 'social_poverty',
                'prompt': 'The poor people of London',
                'expected_keywords': ['poverty', 'poor', 'beggars', 'workhouse', 'alms'],
                'expected_phrases': ['poor relief', 'workhouse', 'alms', 'pauper'],
                'context': 'Poverty was widespread in London, with many relying on poor relief and workhouses.',
                'expected_facts': ['poverty', 'poor relief', 'workhouse', 'alms', 'pauper']
            },
            {
                'category': 'tudor_period',
                'prompt': 'During the Tudor period, London',
                'expected_keywords': ['Tudor', 'Henry', 'Elizabeth', 'monarchy', 'Renaissance'],
                'expected_phrases': ['Tudor dynasty', 'Henry VIII', 'Elizabeth I', 'Tudor period'],
                'context': 'The Tudor period (1485-1603) was marked by the reign of the Tudor dynasty.',
                'expected_facts': ['Tudor', '1485', '1603', 'Henry VIII', 'Elizabeth I']
            },
            {
                'category': 'stuart_period',
                'prompt': 'The Stuart period in England',
                'expected_keywords': ['Stuart', 'James', 'Charles', 'monarchy', 'civil war'],
                'expected_phrases': ['Stuart dynasty', 'James I', 'Charles I', 'Civil War'],
                'context': 'The Stuart period (1603-1714) included the English Civil War and Restoration.',
                'expected_facts': ['Stuart', '1603', '1714', 'Civil War', 'Restoration']
            },
            {
                'category': 'georgian_period',
                'prompt': 'During the Georgian period, London',
                'expected_keywords': ['Georgian', 'George', 'Regency', 'enlightenment', 'industrial'],
                'expected_phrases': ['Georgian period', 'George III', 'Regency period', 'Industrial Revolution'],
                'context': 'The Georgian period (1714-1830) saw the Industrial Revolution and Enlightenment.',
                'expected_facts': ['Georgian', '1714', '1830', 'Industrial Revolution', 'Enlightenment']
            }
        ]
        
        # 2. Language Pattern Tests
        language_pattern_tests = [
            {
                'category': 'archaic_pronouns',
                'prompt': 'Methinks the city',
                'expected_keywords': ['thou', 'thee', 'thy', 'thine', 'hast', 'hath', 'doth', 'dost', 'art', 'wilt', 'shalt'],
                'expected_phrases': ['methinks', 'verily', 'indeed', 'forsooth'],
                'context': 'Historical language patterns from 1500-1850 period.',
                'expected_facts': ['archaic pronouns', 'historical language', 'Early Modern English']
            },
            {
                'category': 'archaic_adverbs',
                'prompt': 'Verily, I say unto you',
                'expected_keywords': ['verily', 'indeed', 'forsooth', 'methinks', 'perchance', 'albeit'],
                'expected_phrases': ['verily', 'I say unto you', 'forsooth'],
                'context': 'Archaic adverbs and expressions common in historical texts.',
                'expected_facts': ['archaic adverbs', 'historical expressions', 'Early Modern English']
            },
            {
                'category': 'archaic_prepositions',
                'prompt': 'Betwixt the houses',
                'expected_keywords': ['betwixt', 'amongst', 'amidst', 'anon', 'ere', 'whilst'],
                'expected_phrases': ['betwixt', 'amongst', 'amidst'],
                'context': 'Archaic prepositions and conjunctions from historical period.',
                'expected_facts': ['archaic prepositions', 'historical conjunctions', 'Early Modern English']
            },
            {
                'category': 'archaic_interjections',
                'prompt': 'Prithee, good sir',
                'expected_keywords': ['prithee', 'pray thee', 'I pray you', 'beseech', 'ye', 'yon'],
                'expected_phrases': ['prithee', 'pray thee', 'I pray you'],
                'context': 'Archaic interjections and polite expressions from historical period.',
                'expected_facts': ['archaic interjections', 'polite expressions', 'historical language']
            }
        ]
        
        # 3. London-Specific Tests
        london_specific_tests = [
            {
                'category': 'london_geography',
                'prompt': 'The River Thames flows through',
                'expected_keywords': ['Thames', 'London', 'river', 'water', 'flow'],
                'expected_phrases': ['River Thames', 'through London', 'Thames River'],
                'context': 'The River Thames is the main river flowing through London.',
                'expected_facts': ['Thames', 'London', 'river', 'geography']
            },
            {
                'category': 'london_landmarks',
                'prompt': 'St. Paul\'s Cathedral is located in',
                'expected_keywords': ['St. Paul\'s', 'Cathedral', 'London', 'City', 'Ludgate'],
                'expected_phrases': ['St. Paul\'s Cathedral', 'in London', 'City of London'],
                'context': 'St. Paul\'s Cathedral is a major landmark in the City of London.',
                'expected_facts': ['St. Paul\'s Cathedral', 'London', 'landmark', 'City of London']
            },
            {
                'category': 'london_areas',
                'prompt': 'Westminster is known for',
                'expected_keywords': ['Westminster', 'Parliament', 'Abbey', 'government', 'royal'],
                'expected_phrases': ['Westminster Abbey', 'Houses of Parliament', 'government'],
                'context': 'Westminster is the political center of London, home to Parliament and Westminster Abbey.',
                'expected_facts': ['Westminster', 'Parliament', 'Abbey', 'government', 'political']
            },
            {
                'category': 'london_markets',
                'prompt': 'Covent Garden was famous for',
                'expected_keywords': ['Covent Garden', 'market', 'flowers', 'vegetables', 'theatre'],
                'expected_phrases': ['Covent Garden market', 'flower market', 'theatre district'],
                'context': 'Covent Garden was a major market area in London, known for flowers and later theatre.',
                'expected_facts': ['Covent Garden', 'market', 'flowers', 'theatre', 'London']
            }
        ]
        
        # 4. Social Class Tests
        social_class_tests = [
            {
                'category': 'nobility',
                'prompt': 'The nobility in London',
                'expected_keywords': ['noble', 'gentleman', 'lady', 'duke', 'earl', 'marquess'],
                'expected_phrases': ['noble class', 'gentleman', 'nobility'],
                'context': 'The nobility were the highest social class in London society.',
                'expected_facts': ['nobility', 'gentleman', 'lady', 'duke', 'earl', 'social class']
            },
            {
                'category': 'merchants',
                'prompt': 'London merchants were',
                'expected_keywords': ['merchant', 'trade', 'commerce', 'guild', 'livery', 'apprentice'],
                'expected_phrases': ['merchant class', 'trade', 'commerce', 'guild'],
                'context': 'Merchants were important in London\'s commercial life and guild system.',
                'expected_facts': ['merchant', 'trade', 'commerce', 'guild', 'livery', 'apprentice']
            },
            {
                'category': 'artisans',
                'prompt': 'London artisans included',
                'expected_keywords': ['artisan', 'craftsman', 'apprentice', 'guild', 'trade', 'skill'],
                'expected_phrases': ['artisan class', 'craftsman', 'apprentice'],
                'context': 'Artisans were skilled craftsmen who worked in various trades in London.',
                'expected_facts': ['artisan', 'craftsman', 'apprentice', 'guild', 'trade', 'skill']
            },
            {
                'category': 'labourers',
                'prompt': 'The working class in London',
                'expected_keywords': ['labourer', 'worker', 'poor', 'poverty', 'wage', 'workhouse'],
                'expected_phrases': ['working class', 'labourer', 'poor', 'workhouse'],
                'context': 'The working class included labourers and poor workers in London.',
                'expected_facts': ['labourer', 'worker', 'poor', 'poverty', 'wage', 'workhouse']
            }
        ]
        
        # 5. Legal and Court Tests
        legal_court_tests = [
            {
                'category': 'old_bailey',
                'prompt': 'The Old Bailey was',
                'expected_keywords': ['Old Bailey', 'court', 'trial', 'judge', 'jury', 'criminal'],
                'expected_phrases': ['Old Bailey', 'criminal court', 'trial', 'judge'],
                'context': 'The Old Bailey was London\'s main criminal court.',
                'expected_facts': ['Old Bailey', 'court', 'trial', 'judge', 'jury', 'criminal']
            },
            {
                'category': 'punishment',
                'prompt': 'Punishments in London included',
                'expected_keywords': ['hanging', 'pillory', 'stocks', 'gaol', 'transport', 'execution'],
                'expected_phrases': ['hanging', 'pillory', 'stocks', 'gaol'],
                'context': 'Various punishments were used in London including hanging, pillory, and gaol.',
                'expected_facts': ['hanging', 'pillory', 'stocks', 'gaol', 'transport', 'execution']
            },
            {
                'category': 'law_enforcement',
                'prompt': 'Law enforcement in London',
                'expected_keywords': ['constable', 'watchman', 'beadle', 'magistrate', 'peace', 'order'],
                'expected_phrases': ['constable', 'watchman', 'beadle', 'magistrate'],
                'context': 'Law enforcement included constables, watchmen, and magistrates.',
                'expected_facts': ['constable', 'watchman', 'beadle', 'magistrate', 'law enforcement']
            }
        ]
        
        # 6. Economic Tests
        economic_tests = [
            {
                'category': 'currency',
                'prompt': 'The currency in London included',
                'expected_keywords': ['shilling', 'pound', 'penny', 'guinea', 'crown', 'farthing'],
                'expected_phrases': ['shilling', 'pound', 'penny', 'guinea'],
                'context': 'London used various coins including shillings, pounds, pennies, and guineas.',
                'expected_facts': ['shilling', 'pound', 'penny', 'guinea', 'crown', 'farthing']
            },
            {
                'category': 'trade',
                'prompt': 'London\'s trade included',
                'expected_keywords': ['trade', 'commerce', 'merchant', 'warehouse', 'port', 'shipping'],
                'expected_phrases': ['trade', 'commerce', 'merchant', 'warehouse'],
                'context': 'London was a major trading center with merchants and warehouses.',
                'expected_facts': ['trade', 'commerce', 'merchant', 'warehouse', 'port', 'shipping']
            },
            {
                'category': 'guilds',
                'prompt': 'London guilds were',
                'expected_keywords': ['guild', 'livery', 'apprentice', 'master', 'craft', 'trade'],
                'expected_phrases': ['guild', 'livery company', 'apprentice', 'master'],
                'context': 'London guilds were trade organizations with apprentices and masters.',
                'expected_facts': ['guild', 'livery', 'apprentice', 'master', 'craft', 'trade']
            }
        ]
        
        # Combine all test cases
        self.test_cases = (
            historical_accuracy_tests + 
            language_pattern_tests + 
            london_specific_tests + 
            social_class_tests + 
            legal_court_tests + 
            economic_tests
        )
    
    def get_test_cases_by_category(self, category: str = None) -> List[Dict[str, Any]]:
        """Get test cases, optionally filtered by category"""
        if category:
            return [tc for tc in self.test_cases if tc['category'] == category]
        return self.test_cases
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(tc['category'] for tc in self.test_cases))
    
    def save_dataset(self, filepath: str = "historical_evaluation_dataset.json"):
        """Save the dataset to a JSON file"""
        dataset = {
            'name': 'London Historical SLM Evaluation Dataset',
            'description': 'Comprehensive evaluation dataset for London Historical SLM (1500-1850)',
            'created': datetime.now().isoformat(),
            'total_test_cases': len(self.test_cases),
            'categories': self.get_categories(),
            'test_cases': self.test_cases
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved to: {filepath}")
        print(f"   Total test cases: {len(self.test_cases)}")
        print(f"   Categories: {len(self.get_categories())}")
    
    def print_summary(self):
        """Print dataset summary"""
        print("\n" + "="*60)
        print("LONDON HISTORICAL EVALUATION DATASET SUMMARY")
        print("="*60)
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"Categories: {len(self.get_categories())}")
        print("\nCategories:")
        for category in sorted(self.get_categories()):
            count = len(self.get_test_cases_by_category(category))
            print(f"  {category}: {count} test cases")
        print("="*60)

def main():
    """Main function to create and save the dataset"""
    print("🏛️ Creating Historical Evaluation Dataset for London Historical SLM")
    print("Period: 1500-1850")
    print("=" * 60)
    
    # Create dataset
    dataset = HistoricalEvaluationDataset()
    
    # Print summary
    dataset.print_summary()
    
    # Save dataset
    dataset.save_dataset()
    
    print("\n✅ Dataset creation completed successfully!")
    print("\nUsage:")
    print("1. Load the dataset in your evaluation script")
    print("2. Use get_test_cases_by_category() to filter by category")
    print("3. Run evaluations on specific historical aspects")

if __name__ == "__main__":
    main()
