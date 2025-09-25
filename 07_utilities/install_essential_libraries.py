#!/usr/bin/env python3
"""
Install Essential Data Processing Libraries
Installs spaCy and NLTK data required for historical_data_collector.py
"""

import subprocess
import sys
import os
from pathlib import Path

def install_spacy_model():
    """Install spaCy English model"""
    print("Installing spaCy English model...")
    print("Note: This may take several minutes (large model ~50MB)...")
    try:
        subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], check=True)
        print("SUCCESS: spaCy English model installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install spaCy model: {e}")
        return False

def install_nltk_data():
    """Install NLTK data"""
    print("Installing NLTK data...")
    print("Note: Downloading punkt, stopwords, and averaged_perceptron_tagger...")
    try:
        subprocess.run([
            sys.executable, "-c", 
            "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
        ], check=True)
        print("SUCCESS: NLTK data installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install NLTK data: {e}")
        return False

def main():
    """Main installation function"""
    print("Installing Essential Data Processing Libraries")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("SUCCESS: Virtual environment detected")
    else:
        print("WARNING: Not in a virtual environment")
        print("   Consider activating your virtual environment first:")
        print("   source activate_env.sh")
        print()
    
    success = True
    
    # Install spaCy model
    if not install_spacy_model():
        success = False
    
    print()
    
    # Install NLTK data
    if not install_nltk_data():
        success = False
    
    print()
    
    if success:
        print("SUCCESS: All essential libraries installed successfully!")
        print("You can now run the data collection pipeline:")
        print("   python 02_data_collection/historical_data_collector.py")
    else:
        print("ERROR: Some installations failed. Check the error messages above.")
        print("You may need to install dependencies manually:")
        print("   pip install spacy nltk")
        print("   python -m spacy download en_core_web_sm")
        print("   python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')\"")

if __name__ == "__main__":
    main()
