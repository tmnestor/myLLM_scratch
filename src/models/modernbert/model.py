"""
ModernBERT model implementation.

This module provides a PyTorch implementation of the ModernBERT architecture,
which differs significantly from standard BERT-like models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernBERTEmbeddings(nn.Module):
    """
    Embeddings module for ModernBERT.
    
    Differs from standard BERT embeddings by:
    1. Not using positional embeddings (handled by rotary embeddings in attention)
    2. Using layer normalization directly after token embeddings
    3. Not using token type embeddings
    """
    
    def __init__(
        self,
        vocab_size=50368,
        hidden_size=768,
        max_position_embeddings=8192,  # Used only for reference, position encoding is in attention
        embedding_dropout=0.0,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # ModernBERT uses a single norm after embeddings, no position embeddings here
        # Position encoding is done using rotary embeddings in the attention layer
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=False)
        self.dropout = nn.Dropout(embedding_dropout)
        
    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        """
        Forward pass through embeddings.
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_length]
            position_ids: Position ids (not used, included for compatibility)
            token_type_ids: Not used in ModernBERT, included for compatibility
            
        Returns:
            Embedded representation of tokens
        """
        # Get token embeddings
        hidden_states = self.tok_embeddings(input_ids)
        
        # Apply normalization and dropout (no position embeddings added)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class ModernBERTRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for ModernBERT.
    
    Implements rotary position embeddings as described in the RoFormer paper.
    This replaces traditional position embeddings and enables better modeling
    of relative distances between tokens.
    """
    
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Initialize the rotation matrix frequencies
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", theta)
        
    def forward(self, x, position_ids):
        """
        Create rotary embeddings for the input.
        
        Args:
            x: Input tensor to rotate
            position_ids: Position indices for each token
            
        Returns:
            tuple of (cos, sin) tensors for applying rotary embeddings
        """
        seq_len = position_ids.shape[-1]
        if seq_len > self.max_position_embeddings:
            # If sequence is longer than max positions, we need to extrapolate
            position_ids = position_ids[:, :self.max_position_embeddings]
            print(f"Warning: Input sequence length {seq_len} exceeds maximum "
                  f"position embedding length {self.max_position_embeddings}")
            
        # Expand freq to batch size and compute rotations
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies for each position
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = freqs.transpose(1, 2)  # [batch, seq_len, dim/2]
        
        # Create the full embeddings
        emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq_len, dim]
        
        # Convert to sin and cos
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_embedding(q, k, cos, sin):
    """
    Apply rotary position embeddings to queries and keys.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim]
        k: Key tensor of shape [batch, heads, seq_len, head_dim]
        cos: Cosine part of rotary embeddings
        sin: Sine part of rotary embeddings
        
    Returns:
        Rotated query and key tensors
    """
    # Extract dimensions
    batch_size, num_heads, seq_len, head_dim = q.shape
    cos_dim = cos.shape[-1]
    
    # Make sure cos/sin have the right shape
    if cos_dim != head_dim:
        # Adjust dimensions - cos/sin might be half the size if only applied to half the dimensions
        if cos_dim * 2 == head_dim:
            # Repeat each element to match dimension
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        else:
            # Just take what we need
            cos = cos[..., :head_dim]
            sin = sin[..., :head_dim]
    
    # Reshape cos and sin for broadcasting
    cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq, dim]
    
    # Split into even and odd dimensions
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # Reshape cos/sin for proper application
    cos_half = cos[..., 0::2]  # Only use half the dimensions
    sin_half = sin[..., 0::2]  # Only use half the dimensions
    
    # Apply rotation using the rotation matrix
    q_rotate_even = q_even * cos_half - q_odd * sin_half
    q_rotate_odd = q_odd * cos_half + q_even * sin_half
    k_rotate_even = k_even * cos_half - k_odd * sin_half
    k_rotate_odd = k_odd * cos_half + k_even * sin_half
    
    # Interleave the rotated dimensions
    q_rotated = torch.zeros_like(q)
    k_rotated = torch.zeros_like(k)
    q_rotated[..., 0::2] = q_rotate_even
    q_rotated[..., 1::2] = q_rotate_odd
    k_rotated[..., 0::2] = k_rotate_even
    k_rotated[..., 1::2] = k_rotate_odd
    
    return q_rotated, k_rotated


class ModernBERTAttention(nn.Module):
    """
    Attention module for ModernBERT.
    
    Uses a single matrix for query, key, value projections (Wqkv),
    applies rotary position embeddings, and has a separate output projection (Wo).
    """
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        max_position_embeddings=8192,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # Check if hidden size is divisible by num_heads
        if self.head_dim * num_attention_heads != hidden_size:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_attention_heads}")
        
        # Combined QKV projection
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        
        # Output projection
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Rotary position embeddings
        self.rotary_emb = ModernBERTRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings
        )
        
        self.attention_dropout = nn.Dropout(attention_dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass through attention layer with rotary position embeddings.
        
        Args:
            hidden_states: Input of shape [batch_size, seq_length, hidden_size]
            attention_mask: Mask of shape [batch_size, 1, 1, seq_length] (optional)
            position_ids: Position indices for tokens (optional)
            
        Returns:
            Output after attention and output projection
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Generate position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Project to query, key, value
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_length, head_dim]
        
        # Split QKV
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(query, position_ids)
        query, key = apply_rotary_embedding(query, key, cos, sin)
        
        # Scale query
        query = query * (self.head_dim ** -0.5)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape context
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_length, num_heads, head_dim]
        context = context.reshape(batch_size, seq_length, self.hidden_size)
        
        # Apply output projection
        output = self.Wo(context)
        
        return output


class ModernBERTMLP(nn.Module):
    """
    MLP module for ModernBERT.
    
    Uses a Gated Linear Unit (GLU) architecture where:
    - Wi: Projects input to two separate tensors (input and gate)
    - Activation is applied to the input, then multiplied by the gate
    - Wo: Output projection from the gated result
    """
    
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1152,  # The actual intermediate size used in Wo
        mlp_dropout=0.0,
        mlp_bias=False,
    ):
        super().__init__()
        # Input projection produces TWICE the intermediate size (for input and gate)
        self.Wi = nn.Linear(hidden_size, intermediate_size * 2, bias=mlp_bias)
        
        # GELU activation for the input path
        self.act = nn.GELU()
        
        # Dropout applied after gating
        self.drop = nn.Dropout(mlp_dropout)
        
        # Output projection
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)
        
    def forward(self, hidden_states):
        """
        Forward pass using Gated Linear Unit (GLU) activation.
        
        Args:
            hidden_states: Input of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Output after GLU and projection
        """
        # Project to input and gate, then split into two tensors
        input_and_gate = self.Wi(hidden_states)
        input_tensor, gate_tensor = input_and_gate.chunk(2, dim=-1)
        
        # Apply activation to input and multiply by gate
        gated_output = self.act(input_tensor) * gate_tensor
        
        # Apply dropout and project back to hidden size
        output = self.Wo(self.drop(gated_output))
        
        return output


class ModernBERTLayer(nn.Module):
    """
    Layer module for ModernBERT.
    
    Consists of attention, layer normalization, and MLP with a
    different structure than standard transformer layers.
    """
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=1152,  # The actual intermediate size (Wo input dimension)
        attention_dropout=0.0,
        mlp_dropout=0.0,
        mlp_bias=False,
        layer_norm_eps=1e-5,
        max_position_embeddings=8192,
    ):
        super().__init__()
        # Attention with rotary position embeddings
        self.attn = ModernBERTAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
        )
        
        # Layer normalization before attention
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=False)
        
        # MLP with Gated Linear Unit architecture
        self.mlp = ModernBERTMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mlp_dropout=mlp_dropout,
            mlp_bias=mlp_bias,
        )
        
        # Layer normalization before MLP
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=False)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass through layer.
        
        Args:
            hidden_states: Input of shape [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask of shape [batch_size, 1, 1, seq_length]
            position_ids: Position indices for rotary embeddings
            
        Returns:
            Output after layer processing
        """
        # Pre-layer norm for attention
        attn_norm_output = self.attn_norm(hidden_states)
        
        # Attention block with residual connection (pass position_ids for rotary embeddings)
        attn_output = self.attn(attn_norm_output, attention_mask, position_ids)
        hidden_states = hidden_states + attn_output
        
        # Pre-layer norm for MLP
        mlp_norm_output = self.mlp_norm(hidden_states)
        
        # MLP block with residual connection
        mlp_output = self.mlp(mlp_norm_output)
        hidden_states = hidden_states + mlp_output
        
        return hidden_states


class ModernBERTEncoder(nn.Module):
    """
    Encoder module for ModernBERT.
    
    Stacks multiple ModernBERT layers with the option for different
    attention patterns across layers.
    """
    
    def __init__(
        self,
        num_hidden_layers=22,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=1152,  # The actual intermediate size (Wo input dimension)
        attention_dropout=0.0,
        mlp_dropout=0.0,
        mlp_bias=False,
        layer_norm_eps=1e-5,
        max_position_embeddings=8192,
    ):
        super().__init__()
        # Create layers
        self.layers = nn.ModuleList([
            ModernBERTLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
                mlp_bias=mlp_bias,
                layer_norm_eps=layer_norm_eps,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final normalization layer
        self.final_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, bias=False)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass through encoder.
        
        Args:
            hidden_states: Input of shape [batch_size, seq_length, hidden_size]
            attention_mask: Mask of shape [batch_size, 1, 1, seq_length]
            position_ids: Position indices for rotary embeddings
            
        Returns:
            Tuple of output for each layer plus final output after normalization
        """
        all_layer_outputs = []
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
            all_layer_outputs.append(hidden_states)
        
        # Apply final normalization
        output = self.final_norm(hidden_states)
        
        return (output, all_layer_outputs)


class ModernBERTPooler(nn.Module):
    """
    Pooler module for ModernBERT.
    
    Instead of traditional CLS pooling, ModernBERT uses a dedicated pooler
    head with mean pooling and a dense projection.
    """
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5, bias=False)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass through pooler.
        
        Args:
            hidden_states: Input of shape [batch_size, seq_length, hidden_size]
            attention_mask: Mask of shape [batch_size, seq_length]
            
        Returns:
            Pooled representation for each sequence
        """
        # Apply mean pooling with attention mask
        if attention_mask is not None:
            # Convert attention mask to float and unsqueeze last dim
            mask = attention_mask.float().unsqueeze(-1)
            
            # Compute the mean representation while ignoring padding tokens
            pooled_output = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # Simple mean pooling without mask
            pooled_output = hidden_states.mean(dim=1)
        
        # Apply dense projection and normalization
        pooled_output = self.dense(pooled_output)
        pooled_output = self.norm(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        
        return pooled_output


class ModernBERTModel(nn.Module):
    """
    Complete ModernBERT model implementation.
    
    Combines the embeddings, encoder, and pooler components into a full model.
    """
    
    def __init__(
        self,
        vocab_size=50368,
        hidden_size=768,
        num_hidden_layers=22,
        num_attention_heads=12,
        intermediate_size=1152,  # The actual intermediate size (Wo input dimension)
        embedding_dropout=0.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        mlp_bias=False,
        max_position_embeddings=8192,  # Maximum sequence length for position encoding
        layer_norm_eps=1e-5,
        use_pooler=True,
    ):
        super().__init__()
        self.use_pooler = use_pooler
        
        # Embeddings
        self.embeddings = ModernBERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            embedding_dropout=embedding_dropout,
            layer_norm_eps=layer_norm_eps,
        )
        
        # Encoder with rotary position embeddings
        self.encoder = ModernBERTEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            mlp_bias=mlp_bias,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
        )
        
        # Pooler (optional)
        if use_pooler:
            self.pooler = ModernBERTPooler(hidden_size=hidden_size)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_length]
            attention_mask: Attention mask of shape [batch_size, seq_length]
            position_ids: Position ids of shape [batch_size, seq_length]
            token_type_ids: Not used in ModernBERT, included for compatibility
            
        Returns:
            Tuple of:
            - sequence_output: Hidden state for each token
            - pooled_output: Pooled representation for sequence classification
            - all_encoder_layers: List of hidden states from each encoder layer
        """
        # Format attention mask for attention layers
        extended_attention_mask = None
        if attention_mask is not None:
            # Convert attention mask (1 for tokens to attend, 0 for tokens to ignore)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Generate position ids for rotary embeddings if not provided
        if position_ids is None and input_ids is not None:
            # Create position ids
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Process inputs through embeddings (no position embeddings here)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=None,  # No position embeddings in embedding layer
            token_type_ids=token_type_ids,
        )
        
        # Process embeddings through encoder (pass position_ids for rotary embeddings)
        sequence_output, all_encoder_layers = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,  # Pass position ids to encoder for rotary embeddings
        )
        
        # Apply pooling if enabled
        pooled_output = None
        if self.use_pooler:
            pooled_output = self.pooler(sequence_output, attention_mask)
        
        return (sequence_output, pooled_output, all_encoder_layers)


class ModernBERTForSentenceEmbedding(nn.Module):
    """
    ModernBERT model for generating sentence embeddings.
    
    Wrapper around the base ModernBERT model that produces normalized
    sentence embeddings suitable for similarity tasks.
    """
    
    def __init__(
        self,
        vocab_size=50368,
        hidden_size=768,
        num_hidden_layers=22,
        num_attention_heads=12,
        intermediate_size=1152,  # The actual intermediate size (Wo input dimension)
        embedding_dropout=0.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        mlp_bias=False,
        max_position_embeddings=8192,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.model = ModernBERTModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            embedding_dropout=embedding_dropout,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            mlp_bias=mlp_bias,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            use_pooler=True,
        )
        self.hidden_size = hidden_size
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to get sentence embeddings.
        
        Args:
            model_output: Output from model forward pass
            attention_mask: Attention mask of shape [batch_size, seq_length]
            
        Returns:
            Sentence embeddings of shape [batch_size, hidden_size]
        """
        # Get sequence output from model output
        token_embeddings = model_output[0]
        
        # Apply attention mask to get accurate mean
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        """
        Generate normalized sentence embeddings.
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_length]
            attention_mask: Attention mask of shape [batch_size, seq_length]
            
        Returns:
            Normalized sentence embeddings of shape [batch_size, hidden_size]
        """
        # Process through base model
        outputs = self.model(input_ids, attention_mask)
        
        # Use pooled output if available, otherwise do mean pooling
        if outputs[1] is not None:
            sentence_embeddings = outputs[1]  # Use pooler output
        else:
            sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings


def create_modernbert_model(
    max_length=None,
    num_layers=12,  # Using 12 instead of full 22 to reduce computational load
):
    """
    Create a ModernBERT model for sentence embeddings.
    
    Args:
        max_length: Maximum sequence length. Defaults to 8192 (ModernBERT default).
        num_layers: Number of transformer layers. Defaults to 12.
        
    Returns:
        ModernBERTForSentenceEmbedding: A sentence encoder using ModernBERT architecture
    """
    # Use default max_length if None provided
    if max_length is None:
        max_length = 8192
    
    # Create model with appropriate configuration
    model = ModernBERTForSentenceEmbedding(
        vocab_size=50368,
        hidden_size=768,
        num_hidden_layers=num_layers,
        num_attention_heads=12,
        intermediate_size=1152,  # The actual intermediate size (Wo input dimension)
        embedding_dropout=0.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        mlp_bias=False,
        max_position_embeddings=max_length,
        layer_norm_eps=1e-5,
    )
    
    return model