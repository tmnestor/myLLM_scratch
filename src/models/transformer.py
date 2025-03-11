"""
Unified transformer model implementation for sentence embedding generation.

This module provides a simplified, unified PyTorch implementation of transformer
architecture that can be configured for different variants including MiniLM and
ModernBERT through explicit parameters rather than separate model classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import Embeddings, TransformerEncoder


class TransformerModel(nn.Module):
    """
    Unified transformer model that can be configured for different variants
    (MiniLM, ModernBERT, etc.) through explicit parameters.
    """
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,      # Use 384 for MiniLM, 768 for ModernBERT
        num_hidden_layers=12, # Use 3/6/12 for MiniLM variants 
        num_attention_heads=12,
        intermediate_size=3072, # Use 1536 for MiniLM, 3072 for ModernBERT
        dropout_rate=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        max_length=512,
        layer_norm_eps=1e-12,
    ):
        """
        Initialize transformer model with specified configuration.
        
        Args:
            vocab_size (int): Size of the vocabulary. Defaults to 30522 (BERT vocabulary size).
            hidden_size (int): Dimension of hidden layers. Defaults to 768.
            num_hidden_layers (int): Number of transformer layers. Defaults to 12.
            num_attention_heads (int): Number of attention heads. Defaults to 12.
            intermediate_size (int): Size of intermediate feed-forward layers. Defaults to 3072.
            dropout_rate (float): Dropout probability. Defaults to 0.1.
            max_position_embeddings (int): Maximum sequence length for position embeddings. 
                Defaults to 512.
            type_vocab_size (int): Number of token types/segments. Defaults to 2.
            max_length (int): Maximum sequence length for processing. Defaults to 512.
            layer_norm_eps (float): Small constant for layer normalization. Defaults to 1e-12.
        """
        super().__init__()
        
        # Parameter validation
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
            
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        # Create model components
        self.embeddings = Embeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_rate
        )
        self.encoder = TransformerEncoder(
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout_rate,
        )

        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        """
        Run forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Token ids of shape [batch_size, seq_length]
            attention_mask (torch.Tensor, optional): Mask of shape [batch_size, seq_length].
                1 for tokens to attend to, 0 for tokens to ignore. Defaults to None.
            token_type_ids (torch.Tensor, optional): Segment ids of shape [batch_size, seq_length].
                Defaults to None.
            position_ids (torch.Tensor, optional): Position ids of shape [batch_size, seq_length].
                Defaults to None.
                
        Returns:
            tuple:
                - sequence_output (torch.Tensor): Hidden state for each token, shape 
                  [batch_size, seq_length, hidden_size]
                - pooled_output (torch.Tensor): Hidden state for the [CLS] token transformed
                  through a linear layer and tanh activation, shape [batch_size, hidden_size]
                - attention_weights (list): List of attention weights from all layers
        """
        # Enforce max_length by truncating if necessary
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :self.max_length]
            if position_ids is not None:
                position_ids = position_ids[:, :self.max_length]

        # Create attention mask if needed
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # Embedding layer
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

        # Transformer encoder
        encoder_outputs, attention_weights = self.encoder(
            embedding_output, extended_attention_mask
        )
        sequence_output = encoder_outputs[-1]

        # Pool the output for sentence representation - use the first token [CLS]
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.pooler[0](first_token_tensor)  # Linear layer
        pooled_output = self.pooler[1](pooled_output)      # Tanh activation

        return (sequence_output, pooled_output, attention_weights)


class SentenceEncoder(nn.Module):
    """
    Wrapper for transformer models to generate sentence embeddings.
    
    This class produces normalized sentence embeddings suitable for semantic 
    similarity tasks by applying mean pooling over token embeddings and L2 
    normalization.
    """

    def __init__(self, base_model):
        """
        Initialize the sentence encoder with a base transformer model.
        
        Args:
            base_model (TransformerModel): The base transformer model
        """
        super().__init__()
        self.model = base_model

    def mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to produce sentence embeddings.
        
        Calculates the mean of all token embeddings for a sentence, weighted by
        the attention mask to exclude padding tokens.
        
        Args:
            model_output (tuple): Output from transformer model
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_length],
                with 1 for valid tokens and 0 for padding tokens
                
        Returns:
            torch.Tensor: Mean-pooled sentence embeddings of shape [batch_size, hidden_size]
        """
        token_embeddings = model_output[0]  # Get the sequence output (last hidden state)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9  # Avoid division by zero
        )

    def forward(self, input_ids, attention_mask):
        """
        Generate normalized sentence embeddings.
        
        Args:
            input_ids (torch.Tensor): Token ids of shape [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_length],
                with 1 for valid tokens and 0 for padding tokens
                
        Returns:
            torch.Tensor: L2-normalized sentence embeddings of shape [batch_size, hidden_size]
                
        Raises:
            ValueError: If inputs are invalid (None, wrong dimensions, or shape mismatch)
        """
        # Input validation
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        if attention_mask is None:
            raise ValueError("attention_mask cannot be None")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D tensor, got {input_ids.dim()}D")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"attention_mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}")
            
        # Process through model
        outputs = self.model(input_ids, attention_mask)
        
        # Create sentence embeddings through mean pooling
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        
        # Normalize embeddings to unit length (L2 norm) for cosine similarity
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def create_minilm_model(num_layers=6, max_length=None):
    """
    Create a TransformerModel configured for MiniLM.
    
    Args:
        num_layers (int): Number of transformer layers (3, 6, or 12)
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        
    Returns:
        SentenceEncoder: A sentence encoder with MiniLM configuration
    """
    if num_layers not in [3, 6, 12]:
        raise ValueError(f"MiniLM supports 3, 6, or 12 layers, got {num_layers}")
    
    # Use default max_length if None provided
    if max_length is None:
        max_length = 128
    
    model = TransformerModel(
        hidden_size=384,
        num_hidden_layers=num_layers,
        intermediate_size=1536,
        max_length=max_length
    )
    
    return SentenceEncoder(model)


def create_modernbert_model(max_length=None):
    """
    Create a TransformerModel configured for ModernBERT.
    
    Args:
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        
    Returns:
        SentenceEncoder: A sentence encoder with ModernBERT configuration
    """
    # Use default max_length if None provided
    if max_length is None:
        max_length = 512
        
    model = TransformerModel(
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        max_length=max_length
    )
    
    return SentenceEncoder(model)