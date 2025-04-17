import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse

from config import (
    MODEL_CACHE_DIR,
    FINE_TUNE_CONFIG,
    GENERATION_CONFIG,
    FEW_SHOT_DIR
)

class RuleGenerator:
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR
        )
        # Add pad_token if it doesn't exist (common for GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Also update model config if necessary
            self.model_config = AutoModelForCausalLM.from_pretrained(model_name).config
            self.model_config.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR,
            # Pass the updated config if pad_token was added
            config=self.model_config if hasattr(self, 'model_config') else None
        ).to(self.device)
        
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
        # Prepare prompt with few-shot examples if available
        prompt = self._prepare_prompt(text)
        
        # Tokenize input - ensure padding token is set
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=GENERATION_CONFIG.get("max_length", 512)-50).to(self.device) # Leave some space for generation
        
        # Generate rules
        # Ensure model's pad_token_id is set correctly for generation
        gen_kwargs = GENERATION_CONFIG.copy()
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        outputs = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        # Decode and return generated rules
        # Handle potential differences in input/output length due to padding/prompt
        # Decode only the generated part, skipping the prompt
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0, input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def _prepare_prompt(self, text):
        """Prepare the prompt with few-shot examples if available."""
        if not self.few_shot_examples:
            # Updated zero-shot prompt
            return f"""Generate JENA rules that perform semantic checks based on the following input text.

Input Text:
{text}

Generated JENA Rules:"""

        # Build the few-shot prompt
        prompt = "Generate JENA rules that perform semantic checks based on the input text, following the patterns shown in these examples:\n\n"
        prompt += "=" * 20 + " EXAMPLES START " + "=" * 20 + "\n\n"
        for i, example in enumerate(self.few_shot_examples):
            prompt += f"--- Example {i+1} ---\n"
            prompt += f"Input Text:\n{example.get('text', 'Missing text example')}\n\n"
            prompt += f"Generated JENA Rules:\n{example.get('rules', 'Missing rules example')}\n\n"
        prompt += "=" * 20 + " EXAMPLES END " + "=" * 20 + "\n\n"

        # Add the actual task
        prompt += f"--- Task ---\n"
        prompt += f"Input Text:\n{text}\n\n"
        prompt += f"Generated JENA Rules:" # Let the model complete from here
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
    parser.add_argument("--model", type=str, default="gpt2", help="Model name to use")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune the model")
    args = parser.parse_args()
    
    generator = RuleGenerator(args.model)
    
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