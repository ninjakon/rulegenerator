from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model registry to store model configurations
MODEL_REGISTRY = {}


class ModelHandler(ABC):
    """Base abstract class for handling different model architectures."""

    @abstractmethod
    def load_model(self, model_name: str, cache_dir: str, device: str):
        """Load model and tokenizer."""
        pass

    @abstractmethod
    def generate(self, tokenizer, model, inputs: Dict[str, Any], generation_config: Dict[str, Any]) -> str:
        """Generate text from inputs."""
        pass


class CausalLMHandler(ModelHandler):
    """Handler for AutoModelForCausalLM models (GPT-2, CodeGen, etc.)."""

    def load_model(self, model_name: str, cache_dir: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(device)
        return tokenizer, model

    def generate(self, tokenizer, model, inputs: Dict[str, Any], generation_config: Dict[str, Any]) -> str:
        outputs = model.generate(
            **inputs,
            **generation_config
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


class Seq2SeqLMHandler(ModelHandler):
    """Handler for AutoModelForSeq2SeqLM models (T5, BART, etc.)."""

    def load_model(self, model_name: str, cache_dir: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(device)
        return tokenizer, model

    def generate(self, tokenizer, model, inputs: Dict[str, Any], generation_config: Dict[str, Any]) -> str:
        outputs = model.generate(
            **inputs,
            **generation_config
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Register available models


def register_model(model_name: str, handler_class: Type[ModelHandler], config: Optional[Dict[str, Any]] = None):
    """Register a model with its handler and configuration."""
    MODEL_REGISTRY[model_name] = {
        "handler": handler_class(),
        "config": config or {}
    }


# Register default models
register_model("gpt2", CausalLMHandler)
register_model("Salesforce/codegen-350M-multi", CausalLMHandler)
register_model("google-t5/t5-small", Seq2SeqLMHandler)
register_model("google-t5/t5-base", Seq2SeqLMHandler)
register_model("facebook/bart-base", Seq2SeqLMHandler)


def get_model_handler(model_name: str):
    """Get the appropriate model handler for a given model name."""
    if model_name in MODEL_REGISTRY:
        handler = MODEL_REGISTRY[model_name]["handler"]
        logger.info(f"Using registered handler for model: {model_name}")
        return handler
    else:
        logger.warning(
            f"No specific handler found for {model_name}. Defaulting to CausalLM handler.")
        return CausalLMHandler()


def list_models():
    """Print all registered models with their handler types."""
    print("Registered models:")
    for model in MODEL_REGISTRY:
        handler_type = MODEL_REGISTRY[model]["handler"].__class__.__name__
        print(f"  - {model} ({handler_type})")
