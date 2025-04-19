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
from model_handlers import get_model_handler, list_models

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RuleGenerator:
    def __init__(self, model_name, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Get model handler and load model and tokenizer
        self.model_handler = get_model_handler(model_name)
        self.tokenizer, self.model = self.model_handler.load_model(
            model_name,
            MODEL_CACHE_DIR,
            self.device
        )

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
        # Prepare prompt with few-shot examples if available
        prompt = self._prepare_prompt(text)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate rules using the appropriate handler
        return self.model_handler.generate(
            self.tokenizer,
            self.model,
            inputs,
            GENERATION_CONFIG
        )

    def _prepare_prompt(self, text):
        """Prepare the prompt with few-shot examples if available."""
        if not self.few_shot_examples:
            # Basic prompt with better instructions
            return f"""
You are a specialized translator that converts natural language specifications into JENA rules.

JENA rules follow this structure:
1. Prefix declarations (@prefix)
2. Rule declarations with a name in square brackets
3. Condition patterns in the body (above the '->') 
4. Conclusion patterns in the head (after the '->')

Input: {text}

Output JENA rules that capture these requirements. Start with appropriate prefixes and create specific validation rules:
"""

        # Enhanced prompt with few-shot examples and detailed instructions
        prompt = """
You are a specialized translator that converts natural language specifications into formal JENA rules.

JENA rules have specific components:
1. PREFIX declarations (@prefix) that define namespaces
2. Rule declarations with a name in square brackets [ruleName:]
3. Condition patterns (body) that specify what to match
4. Conclusion patterns (head) that specify what to infer or validate
5. The arrow symbol '->' separates conditions from conclusions

Important guidelines:
- Use appropriate namespaces (rdf, spec, xsd, etc.)
- Create descriptive rule names
- Define required checks and validations
- Handle both success (OK) and failure (FAIL) cases
- Use proper JENA syntax for variables (?var), functions, and literals

Here are examples of natural language specifications and their corresponding JENA rules:

"""

        # Add few-shot examples with better formatting
        for example in self.few_shot_examples:
            prompt += f"""
### SPECIFICATION:
{example['text']}

### JENA RULES:
{example['rules']}

"""

        # Add target specification with clear expectations
        prompt += f"""
Now translate the following specification into proper JENA rules:

### SPECIFICATION:
{text}

### JENA RULES:
"""
        return prompt

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
    parser.add_argument("--fine_tune", action="store_true",
                        help="Fine-tune the model")
    parser.add_argument("--list_models", action="store_true",
                        help="List all registered models")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    generator = RuleGenerator(args.model)

    if args.fine_tune:
        # Load training data and fine-tune
        # This is a placeholder - you'll need to implement the data loading
        training_data = []
        generator.fine_tune(training_data)


if __name__ == "__main__":
    main()
