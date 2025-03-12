"""
MiniLM model implementation.

This module exposes the MiniLM model classes and factory function.
"""

from .model import MiniLMModel, MiniLMForSentenceEmbedding, create_minilm_model

__all__ = ['MiniLMModel', 'MiniLMForSentenceEmbedding', 'create_minilm_model']