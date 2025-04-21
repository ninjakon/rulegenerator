import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
import logging
import os

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
    def __init__(self, model_name, prompt_name="default", device=None, token=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.token = token

        # Get model handler and load model and tokenizer
        self.model_handler = get_model_handler(model_name)
        self.tokenizer, self.model = self.model_handler.load_model(
            model_name,
            MODEL_CACHE_DIR,
            self.device,
            token=self.token
        )

        # Get prompt handler
        self.prompt_handler = get_prompt(prompt_name)

        # Load few-shot examples if available
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self):
        """
        Load few-shot examples from subdirectories within the few_shot directory.
        Each subdirectory represents one example and should contain:
        - Exactly one .txt file (for the input text)
        - One or more .jr files (for the expected rules)
        """
        examples = []
        print(f"Looking for few-shot examples in: {FEW_SHOT_DIR}")
        if not FEW_SHOT_DIR.is_dir():
            print(f"Warning: Few-shot directory not found: {FEW_SHOT_DIR}")
            return examples

        # Iterate through items in FEW_SHOT_DIR
        for item in FEW_SHOT_DIR.iterdir():
            if item.is_dir():  # Process only subdirectories
                example_dir = item
                print(f"Processing example directory: {example_dir.name}")
                text_files = list(example_dir.glob("*.txt"))
                rule_files = list(example_dir.glob("*.jr"))

                # Validate structure: exactly one .txt file and at least one .jr file
                if len(text_files) != 1:
                    print(f"Warning: Skipping {example_dir.name}. Found {len(text_files)} .txt files (expected 1).")
                    continue
                if not rule_files:
                    print(f"Warning: Skipping {example_dir.name}. Found no .jr files.")
                    continue

                input_text_file = text_files[0]
                combined_rules = ""

                try:
                    # Read the text file
                    with open(input_text_file, "r", encoding="utf-8") as f:
                        input_text = f.read()

                    # Read and concatenate all rule files
                    for rule_file in sorted(rule_files): # Sort for consistent order
                         with open(rule_file, "r", encoding="utf-8") as f:
                            combined_rules += f.read() + "\n\n" # Add separator between rules

                    # Remove trailing newline/whitespace
                    combined_rules = combined_rules.strip()

                    if input_text and combined_rules:
                         examples.append({"text": input_text, "rules": combined_rules})
                         print(f"  Successfully loaded example: {example_dir.name}")
                    else:
                         print(f"Warning: Skipping {example_dir.name}. Empty text or rules found.")

                except Exception as e:
                    print(f"Warning: Could not load example from {example_dir.name}: {e}")

        if not examples:
            print("Warning: No valid few-shot examples found or loaded.")
        else:
            print(f"Successfully loaded {len(examples)} few-shot examples.")
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
    parser.add_argument("--token", type=str,
                        help="Hugging Face token for accessing gated models")
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

    # If token is provided in command line, use it, otherwise try environment variable
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")

    generator = RuleGenerator(args.model, args.prompt, token=token)

    if args.fine_tune:
        # Load training data and fine-tune
        # This is a placeholder - you'll need to implement the data loading
        print("Fine-tuning requires implementing data loading logic.")
        training_data = [] # Replace with actual data loading
        if training_data:
             generator.fine_tune(training_data)
        else:
             print("No training data provided. Skipping fine-tuning.")


if __name__ == "__main__":
    main()
