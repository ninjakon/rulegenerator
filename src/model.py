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
import json

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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR
        ).to(self.device)
        
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
        
        # Generate rules
        outputs = self.model.generate(
            **inputs,
            **GENERATION_CONFIG
        )
        
        # Decode and return generated rules
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _prepare_prompt(self, text):
        """Prepare the prompt with few-shot examples if available."""
        if not self.few_shot_examples:
            return f"Generate JENA rules for the following text:\n{text}"
        
        prompt = "Here are some examples of text and their corresponding JENA rules:\n\n"
        for example in self.few_shot_examples:
            prompt += f"Text: {example['text']}\nRules: {example['rules']}\n\n"
        prompt += f"Now generate JENA rules for this text:\n{text}"
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
        training_data = []
        generator.fine_tune(training_data)

if __name__ == "__main__":
    main() 