#!/usr/bin/env python3
"""
Command-line script to generate datasets for examples.

This script generates classification datasets and saves them to the data/ directory.
Run this script before running the classification examples.
"""

import os
import sys
from src.data_utils import generate_text_classification_datasets

def main():
    """Generate classification datasets."""
    print("Generating text classification datasets...")
    generate_text_classification_datasets()
    print("Done!")

if __name__ == "__main__":
    main()