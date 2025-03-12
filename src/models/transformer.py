"""
Legacy transformer model class.

This module is deprecated and only kept for backward compatibility.
Use the MiniLM or ModernBERT model classes instead.
"""

import warnings

warnings.warn(
    "The transformer.py module is deprecated. Use MiniLM or ModernBERT model classes instead.",
    DeprecationWarning,
    stacklevel=2
)

from .minilm import create_minilm_model