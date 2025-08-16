"""
Main script for transforming text content using different tones with optional few-shot learning.
"""
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime


from src.preprocess import DataPreprocessor, ToneCategory, FewShotStrategySelector
from src.model_client import ModelClient
from src.prompts import GENERIC_SYSTEM_PROMPT, SIMPLE_TONE_PROMPT, DIRECT_TONE_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tone_transformation.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    if args.tone.lower() not in ['simple', 'direct']:
        raise ValueError(f"Invalid tone '{args.tone}'. Choose 'simple' or 'direct'.")

    if args.num_samples < 1 and args.use_few_shots:
        raise ValueError("Number of samples must be at least 1.")

    logger.info(f"Arguments validated successfully")


def get_input_data(args: argparse.Namespace, preprocessor: DataPreprocessor) -> List[Tuple[str, str]]:
    """Get title and text input, either from args or test data."""
    if args.title and args.text:
        logger.info("Using provided title and text")
        return [(args.title.strip(), args.text.strip())]

    logger.info("No title or text provided. Using default test data.")
    test_data = preprocessor.get_test_data()

    if not test_data:
        raise ValueError("No test data available and no input provided.")

    input_data = [(item.title.strip(), item.text.strip()) for item in test_data]
    return input_data


def build_system_prompt(args: argparse.Namespace, text_length: int) -> Tuple[str, ToneCategory]:
    """Build the system prompt based on tone selection."""
    base_prompt = GENERIC_SYSTEM_PROMPT

    if args.tone.lower() == "simple":
        tone = ToneCategory.SIMPLE
        tone_prompt = SIMPLE_TONE_PROMPT.format(answer_length=text_length)
    elif args.tone.lower() == "direct":
        tone = ToneCategory.DIRECT
        tone_prompt = DIRECT_TONE_PROMPT.format(answer_length=text_length)
    else:
        raise ValueError(f"Unsupported tone: {args.tone}")

    system_prompt = f"{base_prompt}\n\n{tone_prompt}"
    logger.info(f"Built system prompt for {tone.value} tone")

    return system_prompt, tone


def add_few_shot_examples(
        system_prompt: str,
        preprocessor: DataPreprocessor,
        tone: ToneCategory,
        args: argparse.Namespace
) -> str:
    """Add few-shot examples to the system prompt."""
    logger.info(f"Adding {args.num_samples} few-shot examples with {args.few_shot_strategy.value} strategy")

    few_shots = preprocessor.get_few_shot_examples(
        target_tone=tone,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        strategy=args.few_shot_strategy
    )

    if not few_shots:
        logger.warning("No few-shot examples found. Proceeding without examples.")
        return system_prompt + "\n\nNow, please rewrite the following text in the specified tone:\n"

    examples = "\n\nHere are some examples:\n"
    valid_examples = 0

    for idx, (default_entry, transformed_entry) in enumerate(few_shots):
        if not transformed_entry:
            logger.warning(f"Skipping example {idx + 1}: no transformed entry available")
            continue

        examples += f"\nExample {valid_examples + 1}:\n"
        examples += f"Input:\nTitle: {default_entry.title}\nText{default_entry.text}\n"
        examples += f'Output:\n{{"Title": "{transformed_entry.title}", "Text": "{transformed_entry.text}"}}\n'
        valid_examples += 1

    if valid_examples == 0:
        logger.warning("No valid few-shot examples found. Proceeding without examples.")
        return system_prompt + "\n\nNow, please rewrite the following text in the specified tone:\n"

    examples += "\nNow, please rewrite the following text in the specified tone:\n"
    logger.info(f"Successfully added {valid_examples} few-shot examples")

    return system_prompt + examples


def generate_response(system_prompt: str, title: str, text: str) -> str:
    """Generate response using the model client."""
    try:
        model_client = ModelClient()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Title: {title}\nText: {text}"}
        ]

        logger.info("Sending request to model...")
        response = model_client.generate_response(messages)
        logger.info("Successfully received response from model")

        return response.content

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise


def save_response_to_file(results: List, output_path: str) -> None:
    """Save the generated response to a file."""
    path = Path(output_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")
    except IOError as e:
        logger.error(f"Error saving results to file {path}: {e}")
        raise IOError(f"Could not save results to file: {path}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Transform text content using different tones with optional few-shot learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/fewshots.json",
        help="Path to the few-shot examples data file"
    )

    parser.add_argument(
        "--tone",
        type=str,
        default="simple",
        choices=['simple', 'direct'],
        help="Tone for the transformation"
    )

    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Input title for transformation"
    )

    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Input text content for transformation"
    )

    parser.add_argument(
        "--use_few_shots",
        action="store_true",
        help="Include few-shot examples in the prompt"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of few-shot examples to include"
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the few-shot examples"
    )

    parser.add_argument(
        "--few_shot_strategy",
        type=FewShotStrategySelector,
        default=FewShotStrategySelector.BALANCED,
        choices=[fs.value for fs in FewShotStrategySelector],
        help="Strategy for selecting few-shot examples"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser


def main() -> None:
    """Main execution function."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    results = []

    try:
        # Validate arguments
        validate_args(args)

        # Initialize preprocessor
        logger.info(f"Loading data from {args.data_path}")
        preprocessor = DataPreprocessor(args.data_path)

        # Get input data
        inputs = get_input_data(args, preprocessor)

        for title, text in inputs:
            logger.info(f"Processing input: Title='{title}', Text='{text[:50]}...' (truncated for logging)")

            text_length = len(text.split())
            logger.info(f"Processing text with {text_length} words")

            # Build system prompt
            system_prompt, tone = build_system_prompt(args, text_length)

            # Add few-shot examples if requested
            if args.use_few_shots:
                system_prompt = add_few_shot_examples(system_prompt, preprocessor, tone, args)
            else:
                logger.info("Skipping few-shot examples")
                system_prompt += "\n\nNow, please rewrite the following text in the specified tone:\n"

            # Generate and display response
            response = generate_response(system_prompt, title, text)
            results.append(
                {
                    "input": {"title": title, "text": text},
                    "output": json.loads(response)
                }
            )

            print("\n" + "=" * 50)
            print("TRANSFORMATION RESULT")
            print("=" * 50)
            print(response)
            print("=" * 50)

            logger.info("Transformation completed successfully")

        # Save results to file
        output_path = f"./data/{args.tone}_transformed.json"
        # add few_shot to name if also few_shots are used
        if args.use_few_shots:
            output_path = output_path.replace(".json", f"_{args.num_samples}_shots.json")

        # timestamp files to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.replace(".json", f"_{timestamp}.json")
        save_response_to_file(results, output_path)

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
