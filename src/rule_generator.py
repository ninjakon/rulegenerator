import argparse
from pathlib import Path
from model import RuleGenerator
from config import INPUT_DIR, OUTPUT_DIR
from utils import get_timestamp


def process_file(input_file: Path, generator: RuleGenerator, output_dir: Path) -> None:
    """Process a single input file and generate JENA rules."""
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Generate rules
    rules = generator.generate_rules(text)

    # Create output file path
    output_file = output_dir / f"{input_file.stem}.jr"

    # Write rules to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rules)

    print(f"Generated rules for {input_file.name} -> {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate JENA rules from text files")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name to use (default: gpt2)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="default",
        help="Prompt template to use (default: default)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Specific input file to process (optional)"
    )
    args = parser.parse_args()

    # Create timestamp directory structure
    timestamp = get_timestamp()
    model_output_dir = OUTPUT_DIR / args.model / timestamp
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize rule generator
    generator = RuleGenerator(args.model, args.prompt)

    if args.input_file:
        # Process single file
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: File {input_path} not found")
            return
        process_file(input_path, generator, model_output_dir)
    else:
        # Process all files in input directory
        for input_file in INPUT_DIR.glob("*.txt"):
            process_file(input_file, generator, model_output_dir)


if __name__ == "__main__":
    main()
