import logging
from typing import Dict, Any, List, Optional, Type

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompt registry to store prompt templates
PROMPT_REGISTRY = {}


class Prompt:
    """Base class for prompt templates."""
    name = "base"  # Default name for the prompt
    description = "Base prompt template"  # Description of the prompt

    def generate(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a prompt for the given text and optional few-shot examples."""
        raise NotImplementedError("Subclasses must implement generate()")


class DefaultPrompt(Prompt):
    """Default prompt for JENA rules generation with detailed instructions."""
    name = "default"
    description = "Default detailed prompt with JENA rules structure and guidelines"

    def generate(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
        if not few_shot_examples:
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
        for example in few_shot_examples:
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


class SimplePrompt(Prompt):
    """Simple prompt for JENA rules generation with minimal instructions."""
    name = "simple"
    description = "Simple prompt with minimal instructions"

    def generate(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
        prompt = "Generate JENA rules for the following text:\n\n"

        if few_shot_examples:
            for example in few_shot_examples:
                prompt += f"Example Text: {example['text']}\n"
                prompt += f"Example Rules: {example['rules']}\n\n"

        prompt += f"Text: {text}\n"
        prompt += "Rules:"

        return prompt


class StructuredPrompt(Prompt):
    """Highly structured prompt for JENA rules generation with step-by-step approach."""
    name = "structured"
    description = "Structured prompt with step-by-step rule creation process"

    def generate(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
        prompt = """
# JENA Rule Generation Task

Your task is to convert natural language specifications into formal JENA rules.

## Step 1: Identify key entities and properties
- Extract the main objects/concepts mentioned
- Identify constraints, measurements, and requirements
- Note relationships between entities

## Step 2: Define prefix declarations
- Use standard prefixes (rdf, rdfs, xsd)
- Define domain-specific prefixes (myspec, spec)

## Step 3: Create validation rules
- Define check rules with descriptive names
- Implement both success (OK) and failure (FAIL) conditions
- Use proper variable syntax with ?var notation

## Step 4: Format output 
- Follow JENA syntax with proper indentation
- Group related rules together
- Use comments to explain complex logic

"""

        if few_shot_examples:
            prompt += "## Reference examples:\n\n"
            for example in few_shot_examples:
                prompt += f"SPECIFICATION:\n{example['text']}\n\n"
                prompt += f"JENA RULES:\n{example['rules']}\n\n"

        prompt += f"""
## Your task:

SPECIFICATION:
{text}

JENA RULES:
"""

        return prompt


# Register available prompts
def register_prompt(prompt_class: Type[Prompt]):
    """Register a prompt template."""
    prompt_instance = prompt_class()
    PROMPT_REGISTRY[prompt_instance.name] = prompt_instance
    logger.debug(f"Registered prompt: {prompt_instance.name}")


# Register default prompts
register_prompt(DefaultPrompt)
register_prompt(SimplePrompt)
register_prompt(StructuredPrompt)


def get_prompt(prompt_name: str) -> Prompt:
    """Get a prompt handler by name."""
    if prompt_name in PROMPT_REGISTRY:
        logger.info(f"Using registered prompt: {prompt_name}")
        return PROMPT_REGISTRY[prompt_name]
    else:
        logger.warning(
            f"Prompt '{prompt_name}' not found. Using default prompt.")
        return PROMPT_REGISTRY["default"]


def list_prompts():
    """Print all registered prompts with their descriptions."""
    print("Available prompts:")
    for name, prompt in PROMPT_REGISTRY.items():
        print(f"  - {name}: {prompt.description}")
