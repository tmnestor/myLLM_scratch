import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components import Embeddings, TransformerEncoder


class MiniLMModel(nn.Module):
    """MiniLM model implementation using PyTorch components.
    Supports L3, L6, and L12 variants."""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=3,  # Default is 3 for L3 models, use 6 for L6 models or 12 for L12 models
        num_attention_heads=12,
        intermediate_size=1536,
        dropout_rate=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        max_length=128,
    ):
        super().__init__()
        
        # Parameter validation
        if num_hidden_layers not in [3, 6, 12]:
            raise ValueError(f"Unsupported number of layers: {num_hidden_layers}. Must be 3, 6, or 12.")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
            
        self.max_length = max_length
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

        # Pooler with ModuleDict structure to match pretrained
        self.pooler = nn.ModuleDict({
            'dense': nn.Linear(hidden_size, hidden_size),
            'activation': nn.Tanh()
        })

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
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
        pooled_output = self.pooler['dense'](first_token_tensor)
        pooled_output = self.pooler['activation'](pooled_output)

        return (sequence_output, pooled_output, attention_weights)


class SentenceTransformer(nn.Module):
    """Wrapper class for MiniLM to generate sentence embeddings.
    Supports L3, L6, and L12 model variants."""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=3,  # Default is 3 for L3 models, use 6 for L6 models or 12 for L12 models
        num_attention_heads=12,
        intermediate_size=1536,
        max_length=128,
    ):
        super().__init__()
        
        # Parameter validation
        supported_layers = [3, 6, 12]
        if num_hidden_layers not in supported_layers:
            raise ValueError(f"Unsupported number of layers: {num_hidden_layers}. Must be one of {supported_layers}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
            
        self.model = MiniLMModel(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_length=max_length,
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        # Input validation
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        if attention_mask is None:
            raise ValueError("attention_mask cannot be None")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D tensor, got {input_ids.dim()}D")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"attention_mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}")
            
        outputs = self.model(input_ids, attention_mask)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
