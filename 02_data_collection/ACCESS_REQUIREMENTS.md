# Data Source Access Requirements

## Overview
This document outlines the access requirements for all data sources in the Hello London Historical LLM project. Some sources require registration, API keys, or institutional access.

## Access Categories

### 🟢 **No Registration Required (Public Domain/Free Access)**
These sources can be downloaded immediately without any registration:

- **Project Gutenberg Sources** (60+ texts) - All public domain
- **Internet Archive Sources** - Public domain
- **London Lives 1690-1800** - CC-BY-NC (free for research)
- **Old Bailey Online** - Free for research
- **Locating London's Past** - CC-BY-NC (free for research)
- **Source Book of London History** - Public domain
- **Historical Collections of a Citizen of London** - Public domain
- **A Journal of the Plague Year** - Public domain
- **The London Archives** - Free public access
- **Maddison Project** - Academic use (no registration)
- **Demographia London** - Public domain
- **Cambridge Occupations** - Academic use (no registration)
- **London Parish Registers** - Public domain
- **Westminster Parish Records** - Public domain
- **Pepys Diary** - Public domain
- **Evelyn Diary** - Public domain
- **Stow Survey of London** - Public domain
- **Stationers Company Records** - Public domain
- **London Gazette** - Public domain
- **EEBO Texts** - Public domain
- **HathiTrust London** - Public domain

### 🟡 **Free Registration Required**
These sources require free registration but no payment:

- **The National Archives (TNA)** - Fair use policy (free registration)
- **Home Office Records** - Fair use policy (free registration)
- **British History Online** - Academic use (free registration)
- **Connected Histories** - Academic use (free registration)
- **British Library Datasets** - Academic use (free registration)
- **London Chronicles** - Academic use (free registration)
- **London Hearth Tax** - Academic use (free registration)

### 🟠 **Limited Free Access**
These sources have some free content but may require payment for full access:

- **Burney Collection** - Limited free access (full access may require institutional login)

### 🔴 **Subscription/Membership Required**
These sources require paid subscriptions or institutional access:

- **London Newspapers Collection** - Subscription required
- **London Literary Magazines** - Subscription required
- **The London Library Digital Collection** - Library membership required

## Registration Instructions

### The National Archives (TNA)
1. Visit: https://www.nationalarchives.gov.uk/
2. Click "Sign in" or "Register"
3. Create free account with email
4. Access HO series records

### British History Online
1. Visit: https://www.british-history.ac.uk/
2. Click "Register" or "Sign in"
3. Create free account
4. Access digitized books and records

### Connected Histories
1. Visit: https://www.connectedhistories.org/
2. No registration required for basic search
3. Some advanced features may require free account

### British Library
1. Visit: https://bl.iro.bl.uk/
2. Create free account for dataset access
3. Browse collections for London-specific data

## API Keys and Tokens

**None of the current data sources require API keys or tokens.** All sources use direct download URLs or web scraping.

## Recommended Approach

### Phase 1: Start with Free Sources
Begin with the 20+ sources that require no registration:
```bash
cd 02_data_collection
python download_historical_data.py
```

### Phase 2: Register for Free Sources
After Phase 1, register for the free academic sources to access additional data.

### Phase 3: Consider Paid Sources (Optional)
If budget allows, consider subscriptions for newspaper and magazine archives.

## Notes

- **Academic Use**: Most "Academic use" sources are free but may require institutional email
- **Fair Use Policy**: TNA sources are free but have usage guidelines
- **CC-BY-NC**: Creative Commons Non-Commercial - free for research use
- **Public Domain**: No restrictions on use

## Troubleshooting

If downloads fail:
1. Check if registration is required
2. Verify internet connection
3. Check if source is temporarily unavailable
4. Review logs in `london_downloader.log`

## Success Rate Expectations

- **No Registration Sources**: ~90% success rate
- **Free Registration Sources**: ~70% success rate (after registration)
- **Paid Sources**: ~95% success rate (if subscribed)
- **Limited Free Access**: ~30% success rate (limited content available)
