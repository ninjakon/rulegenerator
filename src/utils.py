import json
from pathlib import Path
from typing import Dict, Any

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def validate_jena_rules(rules: str) -> bool:
    """Basic validation of JENA rules format."""
    # Add your JENA rules validation logic here
    # This is a placeholder - implement actual validation
    return True

def format_rules(rules: str) -> str:
    """Format JENA rules for better readability."""
    # Add your rules formatting logic here
    # This is a placeholder - implement actual formatting
    return rules

def create_few_shot_example(text: str, rules: str, output_file: Path) -> None:
    """Create a new few-shot example file."""
    example = {
        "text": text,
        "rules": rules
    }
    save_json_file(example, output_file) 