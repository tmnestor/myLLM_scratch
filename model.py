import torch
import torch.nn as nn
import os
from transformers import AutoModel


class MiniLML3Encoder(nn.Module):
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", max_length=128):
        super(MiniLML3Encoder, self).__init__()
        # Get the environment variable that points to model directory
        model_path = os.environ.get('LLM_MODELS_PATH')
        if not model_path:
            raise EnvironmentError("LLM_MODELS_PATH environment variable is not set")
        
        # Extract just the model name part without organization prefix
        self.model_name = model_name.split('/')[-1] if '/' in model_name else model_name
        local_model_path = os.path.join(model_path, self.model_name)
        
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model folder not found: {local_model_path}")
        
        print(f"Loading model from local path: {local_model_path}")
        # Use local_files_only=True to prevent downloading from internet
        self.transformer = AutoModel.from_pretrained(local_model_path, local_files_only=True)
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take attention mask into account for correct averaging
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, sentences, tokenizer, batch_size=8, device="cuda", normalize_embeddings=True):
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            # Tokenize batch
            encoded_input = tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )

            # Move to specified device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.transformer(**encoded_input)

            # Perform pooling
            embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            # Normalize embeddings
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings
