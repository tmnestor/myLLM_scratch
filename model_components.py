"""
Core model components for MiniLM architecture.

This module implements the foundational building blocks of a transformer-based
language model architecture, including embeddings, attention mechanisms, feed-forward
networks, and the overall transformer structure.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    """Token, position and token type embeddings for transformer model.
    
    Combines three types of embeddings:
    1. Word embeddings (token IDs to vectors)
    2. Position embeddings (token position to vectors)
    3. Token type embeddings (segment/sequence IDs to vectors)
    
    All embeddings are summed together and passed through layer normalization and dropout.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout_rate=0.1,
    ):
        """Initialize embedding layers.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
            hidden_size (int): Dimension of the embedding vectors
            max_position_embeddings (int, optional): Maximum sequence length supported. Defaults to 512.
            type_vocab_size (int, optional): Number of token types/segments. Defaults to 2.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
            
        Raises:
            ValueError: If any of the parameters are invalid
        """
        super().__init__()

        # Validate parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )
        if type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {type_vocab_size}")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """Compute combined embeddings for input tokens.
        
        Args:
            input_ids (torch.Tensor): Token ids of shape [batch_size, seq_length]
            token_type_ids (torch.Tensor, optional): Segment ids of shape [batch_size, seq_length].
                If None, all tokens are assumed to be from the first segment. Defaults to None.
            position_ids (torch.Tensor, optional): Position ids of shape [batch_size, seq_length].
                If None, positions are assigned sequentially starting from 0. Defaults to None.
                
        Returns:
            torch.Tensor: Combined embeddings of shape [batch_size, seq_length, hidden_size]
        """
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for transformer models.
    
    Implements the scaled dot-product attention mechanism that allows the model to
    focus on different parts of the input sequence. The attention is split into
    multiple heads, allowing the model to jointly attend to information from
    different representation subspaces.
    """

    def __init__(self, hidden_size, num_attention_heads, dropout_rate=0.1):
        """Initialize multi-head attention layers.
        
        Args:
            hidden_size (int): Size of hidden dimension
            num_attention_heads (int): Number of attention heads
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
            
        Note:
            hidden_size must be divisible by num_attention_heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Self attention components (matches pretrained model structure)
        self.self = nn.ModuleDict(
            {
                "query": nn.Linear(hidden_size, self.all_head_size),
                "key": nn.Linear(hidden_size, self.all_head_size),
                "value": nn.Linear(hidden_size, self.all_head_size),
            }
        )

        # Output components (matches pretrained model structure)
        self.output = nn.ModuleDict(
            {
                "dense": nn.Linear(hidden_size, hidden_size),
                "LayerNorm": nn.LayerNorm(hidden_size, eps=1e-12),
                "dropout": nn.Dropout(dropout_rate),
            }
        )

    def transpose_for_scores(self, x):
        """Reshape input for multi-head attention computation.
        
        Reshapes the input from [batch_size, seq_length, hidden_size] to
        [batch_size, num_attention_heads, seq_length, attention_head_size]
        for parallel computation of attention across heads.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape 
                [batch_size, num_attention_heads, seq_length, attention_head_size]
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """Compute multi-head attention.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape 
                [batch_size, 1, 1, seq_length]. Values should be 0 for tokens to attend to and 
                large negative values (-10000.0) for tokens to ignore. Defaults to None.
                
        Returns:
            tuple:
                - attention_output (torch.Tensor): Attention output of shape [batch_size, seq_length, hidden_size]
                - attention_probs (torch.Tensor): Attention probabilities of shape 
                    [batch_size, num_attention_heads, seq_length, seq_length]
        """
        # Self attention
        mixed_query_layer = self.self["query"](hidden_states)
        mixed_key_layer = self.self["key"](hidden_states)
        mixed_value_layer = self.self["value"](hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.output["dropout"](attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output processing
        attention_output = self.output["dense"](context_layer)
        attention_output = self.output["dropout"](attention_output)
        attention_output = self.output["LayerNorm"](attention_output + hidden_states)

        return attention_output, attention_probs


class FeedForward(nn.Module):
    """Feed-forward neural network layer in transformer block.
    
    Implements a position-wise feed-forward network with two linear transformations
    and a GELU activation in between, followed by dropout. This is applied to each
    position separately and identically.
    """

    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        """Initialize feed-forward network layers.
        
        Args:
            hidden_size (int): Size of hidden states (input and output dimension)
            intermediate_size (int): Size of intermediate representation 
                (typically 4x hidden_size)
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        # Use nn.GELU() instead of F.gelu to avoid callable issues
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        """Apply feed-forward transformation to input.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer encoder layer.
    
    Implements one layer of a transformer encoder, consisting of a multi-head
    self-attention mechanism followed by a position-wise feed-forward network.
    Each of these sublayers has a residual connection around it and is followed
    by layer normalization.
    """

    def __init__(
        self, hidden_size, num_attention_heads, intermediate_size, dropout_rate=0.1
    ):
        """Initialize transformer layer components.
        
        Args:
            hidden_size (int): Size of hidden dimension
            num_attention_heads (int): Number of attention heads
            intermediate_size (int): Size of intermediate feed-forward layer
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size, num_attention_heads, dropout_rate
        )

        # Intermediate layer (matches pretrained naming)
        self.intermediate = nn.ModuleDict(
            {"dense": nn.Linear(hidden_size, intermediate_size)}
        )

        # Add GELU activation as a module instead of using F.gelu
        self.gelu = nn.GELU()

        # Output layer (matches pretrained naming)
        self.output = nn.ModuleDict(
            {
                "dense": nn.Linear(intermediate_size, hidden_size),
                "LayerNorm": nn.LayerNorm(hidden_size, eps=1e-12),
                "dropout": nn.Dropout(dropout_rate),
            }
        )

    def forward(self, hidden_states, attention_mask=None):
        """Process input through transformer layer.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape 
                [batch_size, 1, 1, seq_length]. Defaults to None.
                
        Returns:
            tuple:
                - layer_output (torch.Tensor): Processed representation of shape 
                    [batch_size, seq_length, hidden_size]
                - attention_weights (torch.Tensor): Attention probabilities of shape
                    [batch_size, num_attention_heads, seq_length, seq_length]
        """
        # Self-attention
        attention_output, attention_weights = self.attention(
            hidden_states, attention_mask
        )

        # Intermediate
        intermediate_output = self.intermediate["dense"](attention_output)
        intermediate_output = self.gelu(intermediate_output)

        # Output
        layer_output = self.output["dense"](intermediate_output)
        layer_output = self.output["dropout"](layer_output)
        layer_output = self.output["LayerNorm"](layer_output + attention_output)

        return layer_output, attention_weights


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers.
    
    The encoder is composed of a stack of identical transformer layers.
    Each layer applies a multi-head self-attention mechanism followed by
    a feed-forward network, with residual connections and layer normalization.
    This implementation supports 3, 6, and 12 layer variants (L3, L6, L12).
    """

    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        dropout_rate=0.1,
    ):
        """Initialize transformer encoder with a stack of transformer layers.
        
        Args:
            num_hidden_layers (int): Number of transformer layers (3, 6, or 12)
            hidden_size (int): Size of hidden dimension
            num_attention_heads (int): Number of attention heads
            intermediate_size (int): Size of intermediate feed-forward layer
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
            
        Raises:
            ValueError: If any of the parameters are invalid
        """
        super().__init__()

        # Validate parameters
        supported_layers = [3, 6, 12]
        if num_hidden_layers not in supported_layers:
            raise ValueError(
                f"Unsupported number of layers: {num_hidden_layers}. Must be one of {supported_layers}"
            )
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, got {num_attention_heads}"
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"
            )
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.layer = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size, num_attention_heads, intermediate_size, dropout_rate
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states, attention_mask=None):
        """Process input through all transformer layers.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape 
                [batch_size, 1, 1, seq_length]. Defaults to None.
                
        Returns:
            tuple:
                - all_encoder_layers (list): List of hidden states from all layers,
                  where each element has shape [batch_size, seq_length, hidden_size]
                - all_attention_weights (list): List of attention weights from all layers,
                  where each element has shape [batch_size, num_attention_heads, seq_length, seq_length]
        """
        all_encoder_layers = []
        all_attention_weights = []

        for layer_module in self.layer:
            hidden_states, attention_weights = layer_module(
                hidden_states, attention_mask
            )
            all_encoder_layers.append(hidden_states)
            all_attention_weights.append(attention_weights)

        return all_encoder_layers, all_attention_weights
