from transformers import AutoTokenizer, AutoConfig
import os

class MiniLMTokenizer:
    """Wrapper around Hugging Face tokenizer for MiniLM models.
    Supports paraphrase-MiniLM-L3-v2, all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2, and all-MiniLM-L12-v2."""
    
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", max_length=128):
        # Validate parameters
        supported_models = [
            "paraphrase-MiniLM-L3-v2",
            "all-MiniLM-L6-v2", 
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L12-v2"
        ]
        if model_name not in supported_models:
            raise ValueError(f"Unsupported model: {model_name}. Must be one of {supported_models}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
            
        # Always use the environment variable path
        model_path = os.environ.get('LLM_MODELS_PATH')
        if not model_path:
            raise EnvironmentError("LLM_MODELS_PATH environment variable is not set")
            
        # Form the full model path
        model_folder = model_name.split('/')[-1] if '/' in model_name else model_name
        local_model_path = os.path.join(model_path, model_folder)
            
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model folder not found: {local_model_path}")
        
        # Load config first to validate
        config = AutoConfig.from_pretrained(local_model_path, local_files_only=True)
        print(f"Loading tokenizer from local path: {local_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        
        # Store max_length and vocab size
        self.max_length = max_length
        self._vocab_size = self.tokenizer.vocab_size
        print(f"Loaded tokenizer with vocabulary size: {self._vocab_size}")
        print(f"Maximum sequence length set to: {self.max_length}")

    def encode(self, sentences, padding=True, truncation=True, return_tensors="pt"):
        """
        Tokenize sentences and prepare them for the model

        Args:
            sentences: Single sentence or list of sentences
            padding: Whether to pad sequences to the longest in the batch
            truncation: Whether to truncate sequences that exceed max length
            return_tensors: Return PyTorch tensors ('pt') or TensorFlow ('tf')

        Returns:
            Dictionary with input_ids, attention_mask, and token_type_ids (if applicable)
        """
        # Validate input
        if sentences is None:
            raise ValueError("sentences cannot be None")
        if not isinstance(sentences, (str, list)):
            raise ValueError(f"sentences must be a string or list, got {type(sentences)}")
        if isinstance(sentences, list) and not all(isinstance(s, str) for s in sentences):
            raise ValueError("All items in sentences list must be strings")
        if return_tensors not in ["pt", "tf", "np"]:
            raise ValueError(f"return_tensors must be 'pt', 'tf', or 'np', got {return_tensors}")
            
        return self.tokenizer(
            sentences,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token ids back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self):
        """Get the vocabulary size for model initialization"""
        return self._vocab_size
