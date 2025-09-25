#!/usr/bin/env python3
"""
Environment Setup for London Historical LLM
Sets up Python environment, installs dependencies, and configures paths
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json

class EnvironmentSetup:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        
        # Environment configuration
        self.config = {
            'python_version': f"{self.python_version.major}.{self.python_version.minor}",
            'system': self.system,
            'project_root': str(self.project_root),
            'data_dir': str(self.project_root / "data"),
            'models_dir': str(self.project_root / "09_models"),
            'logs_dir': str(self.project_root / "logs"),
            'venv_name': "helloLondon"
        }
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print(" Checking Python version...")
        
        if self.python_version < (3, 8):
            print(f"Python {self.python_version.major}.{self.python_version.minor} is not supported")
            print("   Minimum required: Python 3.8")
            return False
        
        print(f"Python {self.python_version.major}.{self.python_version.minor} is compatible")
        return True
    
    def check_system_requirements(self):
        """Check system requirements"""
        print("\n Checking system requirements...")
        
        # Check python3-venv package on Linux systems
        if self.system == "linux":
            if not self._check_python_venv_package():
                print("   ERROR: python3-venv package is required but not installed")
                print("   Please install it with: sudo apt install python3-venv")
                print("   Or for Python 3.12: sudo apt install python3.12-venv")
                return False
            
            # Check python3-dev package for torch.compile support
            if not self._check_python_dev_package():
                print("   WARNING: python3-dev package is not installed")
                print("   This is required for torch.compile optimization")
                print("   Install with: sudo apt install python3-dev")
                print("   Training will work but may be slower without compilation")
        
        # Check available memory
        try:
            if self.system == "windows":
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
            else:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                memory_gb = int(meminfo.split('\n')[0].split()[1]) / (1024**2)
            
            print(f"   RAM: {memory_gb:.1f} GB")
            if memory_gb < 8:
                print("   Warning: Less than 8GB RAM detected")
            else:
                print("   Sufficient RAM")
                
        except Exception as e:
            print(f"   Could not check RAM: {e}")
        
        # Check disk space
        try:
            if self.system == "windows":
                import shutil
                free_space = shutil.disk_usage(self.project_root).free / (1024**3)
            else:
                statvfs = os.statvfs(self.project_root)
                free_space = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            
            print(f"   Disk space: {free_space:.1f} GB available")
            if free_space < 10:
                print("   WARNING:  Warning: Less than 10GB free space")
            else:
                print("    Sufficient disk space")
                
        except Exception as e:
            print(f"   WARNING:  Could not check disk space: {e}")
        
        return True
    
    def _check_python_venv_package(self):
        """Check if python3-venv package is available"""
        try:
            # Try to import ensurepip which is provided by python3-venv
            import ensurepip
            print("   python3-venv package: Available")
            return True
        except ImportError:
            print("   python3-venv package: NOT AVAILABLE")
            return False
    
    def _check_python_dev_package(self):
        """Check if python3-dev package is available"""
        try:
            # Try to import Python.h header (indirectly through distutils)
            import distutils.sysconfig
            import os
            include_dir = distutils.sysconfig.get_python_inc()
            python_h = os.path.join(include_dir, 'Python.h')
            if os.path.exists(python_h):
                print("   python3-dev package: Available")
                return True
            else:
                print("   python3-dev package: NOT AVAILABLE")
                return False
        except Exception:
            print("   python3-dev package: NOT AVAILABLE")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("\nCreating project directories...")
        
        directories = [
            "data",
            "data/london_historical",
            "09_models",
            "09_models/checkpoints",
            "09_models/tokenizers",
            "logs",
            "temp",
            "outputs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"    {directory}")
        
        return True
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        print("\n Creating virtual environment...")
        
        venv_path = self.project_root / self.config['venv_name']
        
        if venv_path.exists():
            print(f"    Virtual environment already exists: {venv_path}")
            # Test if the existing venv is working
            if self._test_virtual_environment(venv_path):
                print(f"    Existing virtual environment is working properly")
                return True
            else:
                print(f"    Existing virtual environment is corrupted, removing and recreating...")
                import shutil
                shutil.rmtree(venv_path)
        
        try:
            if self.system == "windows":
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], check=True, capture_output=True)
            else:
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], check=True, capture_output=True)
            
            print(f"    Virtual environment created: {venv_path}")
            
            # Test the new virtual environment
            if self._test_virtual_environment(venv_path):
                print(f"    Virtual environment is working properly")
                return True
            else:
                print(f"    ERROR: New virtual environment is not working properly")
                return False
            
        except subprocess.CalledProcessError as e:
            print(f"   ERROR: Failed to create virtual environment: {e}")
            return False
    
    def _test_virtual_environment(self, venv_path):
        """Test if virtual environment is working properly"""
        try:
            # Get the path to the virtual environment Python
            venv_python = venv_path / "bin" / "python"
            if self.system == "windows":
                venv_python = venv_path / "Scripts" / "python.exe"
            
            # Test if Python works
            result = subprocess.run([
                str(venv_python), "-c", "import sys; print('Python version:', sys.version)"
            ], check=True, capture_output=True, text=True)
            
            # Test if pip works
            result = subprocess.run([
                str(venv_python), "-m", "pip", "--version"
            ], check=True, capture_output=True, text=True)
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def install_dependencies(self):
        """Install Python dependencies from requirements.txt"""
        print("\nInstalling dependencies...")
        
        # Get the path to the virtual environment Python
        venv_python = self.project_root / self.config['venv_name'] / "bin" / "python"
        if self.system == "windows":
            venv_python = self.project_root / self.config['venv_name'] / "Scripts" / "python.exe"
        
        # First, create the requirements.txt file
        self.create_requirements_file()
        
        # Install from requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            try:
                print(f"   Installing from {requirements_file}...")
                print("   Note: This may take several minutes on slow connections...")
                print("   Installing packages... (Note: This can take a long time depending on your network connection)")
                
                result = subprocess.run([
                    str(venv_python), "-m", "pip", "install", "-r", str(requirements_file),
                    "--progress-bar", "on", "--verbose"
                ], check=True, capture_output=True, text=True)
                
                # Check if installation was actually successful by verifying key packages
                if self._verify_key_packages(venv_python):
                    print(f"    All dependencies installed from requirements.txt")
                    return True
                else:
                    print(f"    Some packages failed to install, trying individual installation...")
                    return self._install_missing_packages(venv_python)
                    
            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to install from requirements.txt: {e}")
                if e.stderr:
                    print(f"   Error details: {e.stderr}")
                print(f"   Falling back to individual package installation...")
                return self._install_dependencies_fallback()
        else:
            print(f"   ERROR: requirements.txt not found, using fallback installation...")
            return self._install_dependencies_fallback()
    
    def _verify_key_packages(self, venv_python):
        """Verify that key packages are installed"""
        # Map package names to their import names
        package_imports = {
            "torch": "torch",
            "transformers": "transformers", 
            "tokenizers": "tokenizers",
            "datasets": "datasets",
            "accelerate": "accelerate",
            "wandb": "wandb",
            "python-dotenv": "dotenv",  # Import name is 'dotenv'
            "requests": "requests",
            "beautifulsoup4": "bs4",    # Import name is 'bs4'
            "lxml": "lxml",
            "pandas": "pandas",
            "numpy": "numpy",
            "spacy": "spacy",
            "nltk": "nltk",
            "langdetect": "langdetect",
            "internetarchive": "internetarchive"
        }
        
        for package_name, import_name in package_imports.items():
            try:
                # Some packages don't have __version__ attribute, so we'll just test import
                subprocess.run([
                    str(venv_python), "-c", f"import {import_name}; print('{package_name} imported successfully')"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                return False
        return True
    
    def _install_missing_packages(self, venv_python):
        """Install only the packages that are missing"""
        print("   Installing missing packages individually...")
        
        # Check which packages are missing
        key_packages = ["torch", "transformers", "tokenizers", "datasets", "accelerate", "wandb", "python-dotenv", "requests", "beautifulsoup4", "lxml", "pandas", "numpy", "spacy", "nltk", "langdetect", "internetarchive"]
        missing_packages = []
        
        for package in key_packages:
            try:
                subprocess.run([
                    str(venv_python), "-c", f"import {package}; print('{package} version:', {package}.__version__)"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                missing_packages.append(package)
        
        if not missing_packages:
            print("    All packages are already installed!")
            return True
        
        print(f"    Missing packages: {', '.join(missing_packages)}")
        
        # Install missing packages
        failed_packages = []
        total_packages = len(missing_packages)
        
        for i, package in enumerate(missing_packages, 1):
            try:
                print(f"   Installing {package}... ({i}/{total_packages})")
                print(f"   This may take a moment...")
                
                subprocess.run([
                    str(venv_python), "-m", "pip", "install", package, 
                    "--progress-bar", "on"
                ], check=True, capture_output=True, text=True)
                print(f"   SUCCESS: {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to install {package}: {e}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n   Warning: {len(failed_packages)} packages still failed to install:")
            for pkg in failed_packages:
                print(f"   - {pkg}")
            print(f"\n   You can try installing them manually later:")
            print(f"   {self.project_root}/{self.config['venv_name']}/bin/pip install {' '.join(failed_packages)}")
            return False
        
        print("    All missing packages installed successfully!")
        return True
    
    def _install_dependencies_fallback(self):
        """Fallback method to install dependencies individually"""
        print("   Using fallback installation method...")
        
        # Core dependencies (fallback list)
        dependencies = [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "tokenizers>=0.12.0",
            "datasets>=2.0.0",
            "accelerate>=0.20.0",
            "wandb>=0.12.0",
            "tqdm>=4.64.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "requests>=2.28.0",
            "beautifulsoup4>=4.11.0",
            "lxml>=4.9.0",
            "python-dotenv>=0.19.0",
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0"
        ]
        
        # System-specific dependencies
        if self.system == "windows":
            dependencies.extend([
                "pywin32>=304",
                "psutil>=5.9.0"
            ])
        else:
            dependencies.extend([
                "psutil>=5.9.0"
            ])
        
        # Install each dependency
        venv_python = self.project_root / self.config['venv_name'] / "bin" / "python"
        if self.system == "windows":
            venv_python = self.project_root / self.config['venv_name'] / "Scripts" / "python.exe"
        
        failed_packages = []
        for dep in dependencies:
            try:
                print(f"   Installing {dep}...")
                result = subprocess.run([
                    str(venv_python), "-m", "pip", "install", dep
                ], check=True, capture_output=True, text=True)
                print(f"    {dep}")
            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to install {dep}: {e}")
                if e.stderr:
                    print(f"   Error details: {e.stderr}")
                failed_packages.append(dep)
        
        if failed_packages:
            print(f"\n   Warning: {len(failed_packages)} packages failed to install:")
            for pkg in failed_packages:
                print(f"   - {pkg}")
            print(f"\n   You can try installing them manually later:")
            print(f"   {self.project_root}/{self.config['venv_name']}/bin/pip install {' '.join(failed_packages)}")
            return False
        
        return True
    
    def verify_installation(self):
        """Verify that all required packages are installed"""
        print("\n Verifying installation...")
        
        # Get the path to the virtual environment Python
        venv_python = self.project_root / self.config['venv_name'] / "bin" / "python"
        if self.system == "windows":
            venv_python = self.project_root / self.config['venv_name'] / "Scripts" / "python.exe"
        
        # Key packages to verify (package name -> import name mapping)
        package_imports = {
            "torch": "torch",
            "transformers": "transformers", 
            "tokenizers": "tokenizers",
            "datasets": "datasets",
            "accelerate": "accelerate",
            "wandb": "wandb",
            "python-dotenv": "dotenv",  # Import name is 'dotenv'
            "requests": "requests",
            "beautifulsoup4": "bs4",    # Import name is 'bs4'
            "lxml": "lxml",
            "pandas": "pandas",
            "numpy": "numpy",
            "spacy": "spacy",
            "nltk": "nltk",
            "langdetect": "langdetect",
            "internetarchive": "internetarchive"
        }
        
        missing_packages = []
        
        for package_name, import_name in package_imports.items():
            try:
                # Some packages don't have __version__ attribute, so we'll just test import
                subprocess.run([
                    str(venv_python), "-c", f"import {import_name}; print('{package_name} imported successfully')"
                ], check=True, capture_output=True)
                print(f"    {package_name}")
            except subprocess.CalledProcessError:
                print(f"   ERROR: {package_name} - NOT INSTALLED")
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"\nWarning: {len(missing_packages)} packages are missing:")
            for pkg in missing_packages:
                print(f"   - {pkg}")
            print(f"\nYou may need to install them manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
        else:
            print(f"\nAll key packages verified successfully!")
            return True
    
    def setup_evaluation_data(self):
        """Download required data models for evaluation"""
        print("\nSetting up evaluation data models...")
        
        # Get the path to the virtual environment Python
        venv_python = self.project_root / self.config['venv_name'] / "bin" / "python"
        if self.system == "windows":
            venv_python = self.project_root / self.config['venv_name'] / "Scripts" / "python.exe"
        
        # Download spaCy English model
        try:
            print("   Downloading spaCy English model...")
            print("   Note: This may take several minutes (large model ~50MB)...")
            subprocess.run([
                str(venv_python), "-m", "spacy", "download", "en_core_web_sm"
            ], check=True, capture_output=True)
            print("   SUCCESS: spaCy English model downloaded successfully")
        except subprocess.CalledProcessError as e:
            print("   WARNING: spaCy model download failed.")
            print("   This is ESSENTIAL for data cleaning. Install manually after activation:")
            print("   source activate_env.sh")
            print("   python -m spacy download en_core_web_sm")
        
        # Download NLTK data
        try:
            print("   Downloading NLTK data...")
            print("   Note: Downloading punkt, stopwords, and averaged_perceptron_tagger...")
            subprocess.run([
                str(venv_python), "-c", 
                "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
            ], check=True, capture_output=True)
            print("   SUCCESS: NLTK data downloaded successfully")
        except subprocess.CalledProcessError as e:
            print("   WARNING: NLTK data download failed.")
            print("   This is ESSENTIAL for data cleaning. Install manually after activation:")
            print("   source activate_env.sh")
            print("   python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')\"")
        
        return True
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        print("\n Creating requirements.txt...")
        
        requirements_content = """# London Historical LLM Requirements
# Core ML libraries
torch>=1.9.0
transformers>=4.20.0
tokenizers>=0.12.0
datasets>=2.0.0
accelerate>=0.20.0

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.1.0

# Text processing and NLP
spacy>=3.4.0
nltk>=3.7.0
langdetect>=1.0.9

# Web scraping and data collection
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
internetarchive>=3.0.0

# Visualization and monitoring
matplotlib>=3.5.0
seaborn>=0.11.0
wandb>=0.12.0

# Development tools
jupyter>=1.0.0
ipywidgets>=7.7.0
tqdm>=4.64.0

# System utilities
psutil>=5.9.0
"""
        
        if self.system == "windows":
            requirements_content += "\n# Windows specific\npywin32>=304\n"
        
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        print(f"    Created: {requirements_file}")
        return True
    
    def create_environment_script(self):
        """Create environment activation script"""
        print("\n Creating environment activation script...")
        
        if self.system == "windows":
            script_content = f"""@echo off
REM Hello London Environment Activation
echo Hello London - Activating Environment...

REM Activate virtual environment
call "{self.project_root}\\{self.config['venv_name']}\\Scripts\\activate.bat"

REM Set environment variables
set HELLO_LONDON_ROOT={self.project_root}
set HELLO_LONDON_DATA={self.config['data_dir']}
set HELLO_LONDON_MODELS={self.config['models_dir']}

echo Environment activated!
echo 📁 Project root: %HELLO_LONDON_ROOT%
echo  Data directory: %HELLO_LONDON_DATA%
echo 🤖 Models directory: %HELLO_LONDON_MODELS%
echo.
echo Ready to start training your Hello London LLM!"""
            script_file = self.project_root / "activate_env.bat"
        else:
            script_content = f"""#!/bin/bash
# Hello London Environment Activation
echo "Hello London - Activating Environment..."

# Activate virtual environment
source "{self.project_root}/{self.config['venv_name']}/bin/activate"

# Set environment variables
export HELLO_LONDON_ROOT="{self.project_root}"
export HELLO_LONDON_DATA="{self.config['data_dir']}"
export HELLO_LONDON_MODELS="{self.config['models_dir']}"

echo "Environment activated!"
echo "📁 Project root: $HELLO_LONDON_ROOT"
echo " Data directory: $HELLO_LONDON_DATA"
echo "🤖 Models directory: $HELLO_LONDON_MODELS"
echo ""
echo "Ready to start training your Hello London LLM!"
"""
            script_file = self.project_root / "activate_env.sh"
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        if self.system != "windows":
            os.chmod(script_file, 0o755)
        
        print(f"    Created: {script_file}")
        return True
    
    def save_config(self):
        """Save environment configuration"""
        print("\n Saving environment configuration...")
        
        config_file = self.project_root / "environment_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"    Configuration saved: {config_file}")
        return True
    
    def print_activation_instructions(self):
        """Print instructions for activating the environment"""
        print("\n" + "="*70)
        print(" ENVIRONMENT SETUP COMPLETE!")
        print("="*70)
        print(f"Project root: {self.project_root}")
        print(f" Python version: {self.config['python_version']}")
        print(f" System: {self.config['system']}")
        print(f"Virtual environment: {self.config['venv_name']}")
        
        print(f"\n To activate the environment:")
        if self.system == "windows":
            print(f"   {self.project_root}\\activate_env.bat")
            print(f"   OR")
            print(f"   {self.project_root}\\{self.config['venv_name']}\\Scripts\\activate")
        else:
            print(f"   source {self.project_root}/activate_env.sh")
            print(f"   OR")
            print(f"   source {self.project_root}/{self.config['venv_name']}/bin/activate")
        
        print(f"\n Next steps:")
        print(f"   1. Activate the environment")
        print(f"   2. If spaCy/NLTK failed, install manually:")
        print(f"      python -m spacy download en_core_web_sm")
        print(f"      python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')\"")
        print(f"   3. Run: cd 02_data_collection && python historical_data_collector.py")
        print(f"   4. Run: cd 03_tokenizer && python train_historical_tokenizer.py")
        print(f"   5. Run: cd 04_training && python train_model_slm.py")
        print(f"   6. Run: cd 05_evaluation && python run_evaluation.py --mode quick")
        
        print("="*70)

def main():
    """Main setup function"""
    print("Hello London - Environment Setup")
    print("=" * 50)
    
    setup = EnvironmentSetup()
    
    # Run setup steps
    steps = [
        ("Checking Python version", setup.check_python_version),
        ("Checking system requirements", setup.check_system_requirements),
        ("Creating directories", setup.create_directories),
        ("Creating virtual environment", setup.create_virtual_environment),
        ("Installing dependencies", setup.install_dependencies),
        ("Verifying installation", setup.verify_installation),
        ("Setting up evaluation data", setup.setup_evaluation_data),
        ("Creating activation script", setup.create_environment_script),
        ("Saving configuration", setup.save_config)
    ]
    
    total_steps = len(steps)
    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n[{i}/{total_steps}] {step_name}...")
        if not step_func():
            print(f"Setup failed at: {step_name}")
            return False
    
    # Print final instructions
    setup.print_activation_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
