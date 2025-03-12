"""
Simplified tokenizer implementation for transformer models.

This module provides a clean, straightforward tokenizer implementation that
works with HuggingFace tokenizers without unnecessary abstractions.
"""

import os
from transformers import AutoTokenizer
import torch


class Tokenizer:
    """
    Tokenizer for transformer models that handles tokenization for inference.
    
    This class wraps HuggingFace tokenizers and provides convenient methods
    for encoding text inputs in the format expected by transformer models.
    """
    
    def __init__(
        self,
        model_path,
        max_length=128,
        padding="max_length",
        truncation=True
    ):
        """
        Initialize tokenizer from a model path.
        
        Args:
            model_path (str): Path to the pretrained tokenizer or model name
            max_length (int): Maximum sequence length
            padding (str/bool): Padding strategy ('max_length', 'longest', or True/False)
            truncation (bool): Whether to truncate sequences to max_length
            
        Raises:
            ValueError: If model_path is invalid or tokenizer cannot be loaded
        """
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Check if local path exists
        if os.path.isdir(model_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"Loaded tokenizer from local path: {model_path}")
            except Exception as e:
                raise ValueError(f"Failed to load tokenizer from {model_path}: {e}")
        # Otherwise, try to load from HuggingFace
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"Loaded tokenizer from HuggingFace: {model_path}")
            except Exception as e:
                # Try to use environment variable fallback
                env_path = os.environ.get("LLM_MODELS_PATH")
                if env_path:
                    full_path = os.path.join(env_path, model_path)
                    if os.path.isdir(full_path):
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(full_path)
                            print(f"Loaded tokenizer from LLM_MODELS_PATH: {full_path}")
                        except Exception as e_env:
                            raise ValueError(f"Failed to load tokenizer from {full_path}: {e_env}")
                    else:
                        raise ValueError(f"Model path not found: {model_path} or {full_path}")
                else:
                    raise ValueError(f"Failed to load tokenizer {model_path} and LLM_MODELS_PATH not set: {e}")
        
        # Report vocabulary size and max length
        vocab_size = len(self.tokenizer.get_vocab())
        print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
        print(f"Maximum sequence length set to: {max_length}")
    
    def encode(self, texts, return_tensors="pt"):
        """
        Encode a list of texts into model input format.
        
        Args:
            texts (list/str): Text or list of texts to encode
            return_tensors (str): Return format ('pt' for PyTorch, 'tf' for TensorFlow)
            
        Returns:
            dict: Dictionary with input_ids and attention_mask as tensors
                - input_ids: Token IDs
                - attention_mask: Attention mask (1 for real tokens, 0 for padding)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors
        )
        
        return encoded


def get_tokenizer_for_model(model_name, max_length=None):
    """
    Helper function to get an appropriate tokenizer for a standard model.
    
    Args:
        model_name (str): Name of the model ('minilm-l3', 'minilm-l6', 'minilm-l12', 'modernbert')
        max_length (int): Override max_length (optional)
        
    Returns:
        Tokenizer: Configured tokenizer for the model
    """
    # Set default paths based on model name
    if model_name == "minilm-l3":
        path = "paraphrase-MiniLM-L3-v2"
        default_length = 128
    elif model_name == "minilm-l6":
        path = "all-MiniLM-L6-v2" 
        default_length = 128
    elif model_name == "minilm-l12":
        path = "all-MiniLM-L12-v2"
        default_length = 128
    elif model_name == "modernbert":
        path = "ModernBERT-base"
        default_length = 512
    else:
        # For direct paths
        path = model_name
        default_length = 128
    
    # Use provided max_length or default
    max_length = max_length or default_length
    
    return Tokenizer(path, max_length=max_length)