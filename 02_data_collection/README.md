# Data Collection Directory

This directory contains the unified data collection and processing system for the London Historical LLM project.

> **📚 For complete documentation**: See [Data Collection Guide](08_documentation/DATA_COLLECTION.md) and [Training Quick Start](08_documentation/TRAINING_QUICK_START.md)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data collection
python historical_data_collector.py
```

## Key Files

- **`historical_data_collector.py`** - Main data collection system
- **`data_sources.json`** - Data source definitions
- **`ACCESS_REQUIREMENTS.md`** - Detailed access requirements for all sources

## Utilities

- **`count_tokens.py`** - Token counting utility
- **`sanitize_filenames.py`** - Filename sanitization utility
- **`synthetic_data_generator.py`** - Synthetic data generation

## Access Requirements

See [ACCESS_REQUIREMENTS.md](ACCESS_REQUIREMENTS.md) for detailed information about:
- Data sources requiring registration
- Free vs. paid sources
- Step-by-step registration instructions
- Expected success rates

## Troubleshooting

For detailed troubleshooting, see the [Data Collection Guide](08_documentation/DATA_COLLECTION.md).