Ape-ING-2025

Code for the master thesis “Gemini-Driven Automated Prompt Engineering for Mining High-Volume Sustainability Reports at ING Bank.”
Author: Vitalii Fishchuk, University of Twente (EEMCS)

Thesis link: https://purl.utwente.nl/essays/108732

Overview

This repository contains the workflow and example code for automated prompt engineering using Gemini to mine large volumes of sustainability reports. It focuses on building an optimization loop that iteratively refines prompts for higher-quality extractions.

Quickstart
# 1) Create & activate a virtual environment (example for bash)
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the example optimization workflow
python notebooks/example_usage.py
