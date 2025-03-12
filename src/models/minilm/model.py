"""
MiniLM model implementation.

This module provides a PyTorch implementation of the MiniLM architecture,
which is a lightweight version of BERT with a standard transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components import Embeddings, TransformerEncoder


class MiniLMModel(nn.Module):
    """
    Complete MiniLM model implementation.
    
    Combines the embeddings, encoder, and pooler components with the correct
    architecture and dimensions for MiniLM models.
    """
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        dropout_rate=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        max_length=128,
        layer_norm_eps=1e-12,
    ):
        """
        Initialize MiniLM model with specified configuration.
        
        Args:
            vocab_size (int): Size of the vocabulary. Defaults to 30522 (BERT vocabulary size).
            hidden_size (int): Dimension of hidden layers. Defaults to 384 for MiniLM.
            num_hidden_layers (int): Number of transformer layers (3, 6, or 12). Defaults to 6.
            num_attention_heads (int): Number of attention heads. Defaults to 12.
            intermediate_size (int): Size of intermediate feed-forward layers. Defaults to 1536.
            dropout_rate (float): Dropout probability. Defaults to 0.1.
            max_position_embeddings (int): Maximum sequence length for position embeddings. 
                Defaults to 512.
            type_vocab_size (int): Number of token types/segments. Defaults to 2.
            max_length (int): Maximum sequence length for processing. Defaults to 128.
            layer_norm_eps (float): Small constant for layer normalization. Defaults to 1e-12.
        """
        super().__init__()
        
        # Parameter validation
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_hidden_layers not in [3, 6, 12]:
            raise ValueError(f"MiniLM supports num_hidden_layers of 3, 6, or 12, got {num_hidden_layers}")
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

        # Pooler (for [CLS] token)
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


class MiniLMForSentenceEmbedding(nn.Module):
    """
    MiniLM model for generating sentence embeddings.
    
    Wrapper around the base MiniLM model that produces normalized
    sentence embeddings suitable for similarity tasks.
    """

    def __init__(self, base_model):
        """
        Initialize the sentence encoder with a base MiniLM model.
        
        Args:
            base_model (MiniLMModel): The base MiniLM model
        """
        super().__init__()
        self.model = base_model
        self.hidden_size = base_model.hidden_size

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
    Create a MiniLM model for sentence embeddings.
    
    Args:
        num_layers (int): Number of transformer layers (3, 6, or 12)
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        
    Returns:
        MiniLMForSentenceEmbedding: A sentence encoder using MiniLM architecture
    """
    if num_layers not in [3, 6, 12]:
        raise ValueError(f"MiniLM supports 3, 6, or 12 layers, got {num_layers}")
    
    # Use default max_length if None provided
    if max_length is None:
        max_length = 128
    
    # Create base model with correct MiniLM configuration
    base_model = MiniLMModel(
        hidden_size=384,
        num_hidden_layers=num_layers,
        intermediate_size=1536,
        max_length=max_length
    )
    
    # Wrap with sentence embedding model
    return MiniLMForSentenceEmbedding(base_model)