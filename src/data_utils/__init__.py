"""
Data utilities for dataset generation and preprocessing.

This module provides utilities for generating, loading, and preprocessing
datasets for text classification and similarity examples.
"""

from .generator import generate_text_classification_datasets

__all__ = ["generate_text_classification_datasets"]