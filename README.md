# JENA Rule Generator

This project uses Large Language Models (LLMs) to generate JENA rules from clear text files.

## Project Structure

```
.
├── data/
│   ├── input/              # Clear text files to process
│   ├── output/             # Generated JENA rules
│   └── few_shot/           # Few-shot examples for LLM prompting
├── src/
│   ├── cleanup.py          # Utility to cleanup old outputs
│   ├── config.py           # Configuration settings
│   ├── model.py            # LLM handling and fine-tuning
│   ├── model_handlers.py   # Model architecture handlers
│   ├── rule_generator.py   # Main rule generation logic
│   └── utils.py            # Utility functions
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your clear text files in the `data/input/` directory
2. (Optional) Add few-shot examples in `data/few_shot/` directory
3. Run the rule generator:

```bash
python src/rule_generator.py --model "gpt2" --input_file "your_file.txt"
```

Output files will be organized by model name and timestamp in the `data/output/` directory.

## Configuration

- Model selection can be done via command line arguments
- Few-shot examples can be added in the `data/few_shot/` directory
- Fine-tuning configurations can be modified in `src/config.py`

## Model Support

The rule generator supports multiple model architectures through a flexible handler system:

### Pre-registered Models

The following models are pre-registered and ready to use:

- **CausalLM models** (GPT-style, auto-regressive):

  - `gpt2` - OpenAI GPT-2
  - `Salesforce/codegen-350M-multi` - CodeGen for multi-language code generation

- **Seq2SeqLM models** (Encoder-Decoder, T5-style):
  - `google-t5/t5-small` - Google's T5 small model
  - `google-t5/t5-base` - Google's T5 base model
  - `facebook/bart-base` - Facebook's BART base model

### Listing Available Models

To see all registered models and their handler types:

```bash
python src/model.py --list_models
```

### Custom Model Support

You can add support for additional models by modifying `src/model_handlers.py`:

1. Create a new handler class if needed (for different model architectures)
2. Register your model using the `register_model` function

Example of registering a new model:

```python
# In src/model_handlers.py
register_model("your/custom-model", CausalLMHandler)
```

## Fine-tuning

To fine-tune the model on your specific use case:

1. Prepare your training data
2. Modify fine-tuning parameters in `src/config.py`
3. Run the fine-tuning script:

```bash
python src/model.py --fine_tune
```

## Cleanup

To clean up old output directories:

```bash
python src/cleanup.py --delete-old-outputs
```

Or use the shorter option:

```bash
python src/cleanup.py -doo
```

This will remove all timestamp directories older than the current time while preserving the model directory structure.
