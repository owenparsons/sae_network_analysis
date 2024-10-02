# Feature Interactions in Language Models

This repository contains code used for the research project found here: [Exploring Feature Co-Occurrence Networks with SAEs](https://owenparsons.github.io/saes/).

This was done as a final project for the [BlueDot AI Alignment course](https://aisafetyfundamentals.com/alignment/)

## Project Description

The project focused on investigating how we can gain a broader picture of feature interactions in language models by combining current interpretability techniques with network analysis.

I tried to answer three key questions:

1. Can we uncover structure in SAE features by examining co-occurrence patterns, specifically through correlations between feature attribution values?
2. How can we leverage feature steering techniques to gain insights into the dependencies between co-occurring features, or alternatively, to understand how they might contribute independently in similar ways?
3. To what extent can we apply network analysis approaches to assess feature importance and illuminate the relationships between features?

[A demo notebook is available](./notebooks/connectivity_demo.ipynb) to showcase the capabilities of this research.

## Prerequisites

Make sure you have the following installed:

- Python 3.10
- `pip` (Python package installer)
- `virtualenv` (or another env manager, optional but recommended)

## Local Installation

To install this repository as a package locally for development purposes, follow the steps below.

```bash
# Clone the repository to your local machine
git clone https://github.com/owen-parsons/sae_network_analysis.git
cd sae_network_analysis

# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Check the installation
pip list
```

## Libraries Used

This project utilizes the following libraries:
- [SAE Lens](https://github.com/yourusername/sae-lens)
- [Transformer Lens](https://github.com/yourusername/transformer-lens)

## Future Updates

More functionality will be added shortly.

## Last Updated

This repository was last updated on **October 2, 2024**.
