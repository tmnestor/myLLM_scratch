import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components import Embeddings, TransformerEncoder


class MiniLMModel(nn.Module):
    """MiniLM-L3-v2 model implementation using PyTorch components"""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=3,
        num_attention_heads=12,
        intermediate_size=1536,
        dropout_rate=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        max_length=128,
    ):
        super().__init__()
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
    """Wrapper class for MiniLM to generate sentence embeddings"""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=3,
        num_attention_heads=12,
        intermediate_size=1536,
        max_length=128,
    ):
        super().__init__()
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
        outputs = self.model(input_ids, attention_mask)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
