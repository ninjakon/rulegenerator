import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
import json
import logging

from config import (
    MODEL_CACHE_DIR,
    FINE_TUNE_CONFIG,
    GENERATION_CONFIG,
    FEW_SHOT_DIR
)
from model_handlers import get_model_handler, list_models as list_model_handlers
from prompt_handlers import get_prompt, list_prompts

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RuleGenerator:
    def __init__(self, model_name, prompt_name="default", device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.prompt_name = prompt_name

        # Get model handler and load model and tokenizer
        self.model_handler = get_model_handler(model_name)
        self.tokenizer, self.model = self.model_handler.load_model(
            model_name,
            MODEL_CACHE_DIR,
            self.device
        )

        # Get prompt handler
        self.prompt_handler = get_prompt(prompt_name)

        # Load few-shot examples if available
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self):
        """Load few-shot examples from the few_shot directory."""
        examples = []
        for file in FEW_SHOT_DIR.glob("*.json"):
            with open(file, "r") as f:
                examples.extend(json.load(f))
        return examples

    def generate_rules(self, text):
        """Generate JENA rules from input text."""
        # Prepare prompt using the specified prompt handler
        prompt = self.prompt_handler.generate(text, self.few_shot_examples)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate rules using the appropriate handler
        return self.model_handler.generate(
            self.tokenizer,
            self.model,
            inputs,
            GENERATION_CONFIG
        )

    def fine_tune(self, training_data):
        """Fine-tune the model on custom data."""
        # Prepare dataset
        dataset = Dataset.from_dict({
            "text": [item["text"] for item in training_data],
            "rules": [item["rules"] for item in training_data]
        })

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True
        )

        # Setup training arguments
        training_args = TrainingArguments(
            **FINE_TUNE_CONFIG
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(FINE_TUNE_CONFIG["output_dir"])
        self.tokenizer.save_pretrained(FINE_TUNE_CONFIG["output_dir"])


def main():
    parser = argparse.ArgumentParser(description="Rule Generator Model")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name to use")
    parser.add_argument("--prompt", type=str, default="default",
                        help="Prompt template to use")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Fine-tune the model")
    parser.add_argument("--list_models", action="store_true",
                        help="List all registered models")
    parser.add_argument("--list_prompts", action="store_true",
                        help="List all registered prompt templates")
    args = parser.parse_args()

    if args.list_models:
        list_model_handlers()
        return

    if args.list_prompts:
        list_prompts()
        return

    generator = RuleGenerator(args.model, args.prompt)

    if args.fine_tune:
        # Load training data and fine-tune
        # This is a placeholder - you'll need to implement the data loading
        training_data = []
        generator.fine_tune(training_data)


if __name__ == "__main__":
    main()
