import json
import logging
import spacy
import argparse
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, results_path: str = None):
        """
        Initialize the Evaluator with a model client.
        Args:
            results_path (str): Path to the JSON file containing evaluation results.
        """
        self.results_path = results_path
        self.results: List[Dict[str, Any]] = []
        if results_path:
            self.load_results()
            # validate the structure of the results
            self._structured_eval()

    def load_results(self) -> None:
        """
        Load evaluation results from a JSON file.
        """
        path = Path(self.results_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        try:
            with open(path, "r", encoding="utf-8") as file:
                self.results = json.load(file)

        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {self.results_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading results: {e}")

    def _structured_eval(self) -> Dict[str, Any]:
        """
        Analyze the structure of the generated responses.

        Returns:
            A dictionary containing the number of valid entries and total entries.
        """
        if not self.results:
            raise ValueError("No results to analyze.")

        expected_structure = {
            "Title": str,
            "Text": str
        }

        errors = []
        valid_entries = 0

        for idx, result in enumerate(self.results):
            # Check if 'output' key exists
            if "output" not in result:
                errors.append(f"Entry {idx}: Missing 'output' key")
                continue

            output = result["output"]

            # Check if output is a dictionary
            if not isinstance(output, dict):
                errors.append(f"Entry {idx}: 'output' is not a dictionary, got {type(output).__name__}")
                continue

            # Check for missing required keys
            missing_keys = set(expected_structure.keys()) - set(output.keys())
            if missing_keys:
                errors.append(f"Entry {idx}: Missing required keys: {', '.join(missing_keys)}")
                continue

            # Check data types and non-empty values
            entry_valid = True
            for key, expected_type in expected_structure.items():
                if key in output:
                    value = output[key]
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"Entry {idx}: '{key}' expected {expected_type.__name__}, got {type(value).__name__}")
                        entry_valid = False
                    elif expected_type == str and not value.strip():
                        errors.append(f"Entry {idx}: '{key}' is empty or whitespace only")
                        entry_valid = False

            if entry_valid:
                valid_entries += 1

        # Raise error if any validation failed
        if errors:
            error_msg = f"Structure validation failed. {valid_entries}/{len(self.results)} entries valid.\n" + "\n".join(
                errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"All {len(self.results)} entries match expected structure.")
        return {
            "valid_entries": valid_entries,
            "total_entries": len(self.results),
            "structure_compliance": True
        }

    def length_eval(self) -> Dict[str, Any]:
        """
        Analyze the length of the generated responses with comprehensive statistics.

        Returns:
            A dictionary containing detailed length statistics for input and output.

        Raises:
            ValueError: If no results to analyze or if required keys are missing.
        """
        if not self.results:
            raise ValueError("No results to analyze.")

        input_lengths = []
        output_lengths = []
        errors = []

        for idx, result in enumerate(self.results):
            try:
                # Calculate lengths (handle None/empty values)
                input_text = result["input"]["text"] or ""
                output_text = result["output"]["Text"] or ""

                input_word_count = len(input_text.split()) if input_text.strip() else 0
                output_word_count = len(output_text.split()) if output_text.strip() else 0

                input_lengths.append(input_word_count)
                output_lengths.append(output_word_count)

            except Exception as e:
                errors.append(f"Entry {idx}: Error processing lengths - {str(e)}")

        if errors:
            logger.warning(f"Length analysis warnings: {'; '.join(errors)}")

        if not input_lengths or not output_lengths:
            raise ValueError("No valid entries found for length analysis.")

        # Calculate comprehensive statistics for both input and output
        return {
            # Input statistics
            "average_input_length": sum(input_lengths) / len(input_lengths),
            "max_input_length": max(input_lengths),
            "min_input_length": min(input_lengths),

            # Output statistics
            "average_output_length": sum(output_lengths) / len(output_lengths),
            "max_output_length": max(output_lengths),
            "min_output_length": min(output_lengths),

            # Additional metrics
            "total_entries_analyzed": len(input_lengths),
            "entries_with_errors": len(errors),
            "average_length_ratio": (sum(output_lengths) / sum(input_lengths)) if sum(input_lengths) > 0 else 0.0
        }

    def similarity_eval(self, similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Evaluate the similarity of generated responses to expected outputs with enhanced metrics.

        Args:
            similarity_threshold: Threshold for considering text as similar (default: 0.8)

        Returns:
            A dictionary containing similarity metrics and match percentages.

        Raises:
            ValueError: If no results to evaluate or spaCy model loading fails.
        """
        if not self.results:
            raise ValueError("No results to evaluate similarity.")

        try:
            # need to use large model for similarity
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise ValueError(
                "spaCy English model 'en_core_web_lg' not found. Install with: python -m spacy download en_core_web_lg")

        title_matches = 0
        text_matches = 0
        similarity_scores = []
        errors = []
        valid_entries = 0

        for idx, result in enumerate(self.results):
            try:
                input_data = result["input"]
                output_data = result["output"]

                # Check required fields
                if "title" not in input_data or "Title" not in output_data:
                    errors.append(f"Entry {idx}: Missing title fields")
                    continue

                if "text" not in input_data or "Text" not in output_data:
                    errors.append(f"Entry {idx}: Missing text fields")
                    continue

                # Title comparison (case-insensitive, strip whitespace)
                input_title = (input_data["title"] or "").strip().lower()
                output_title = (output_data["Title"] or "").strip().lower()

                if input_title and output_title and input_title == output_title:
                    title_matches += 1

                # Text similarity using spaCy
                input_text = input_data["text"] or ""
                output_text = output_data["Text"] or ""

                if input_text.strip() and output_text.strip():
                    input_doc = nlp(input_text)
                    output_doc = nlp(output_text)
                    similarity_score = input_doc.similarity(output_doc)
                    similarity_scores.append(similarity_score)

                    if similarity_score > similarity_threshold:
                        text_matches += 1
                else:
                    # Handle empty text cases
                    if not input_text.strip() and not output_text.strip():
                        similarity_scores.append(1.0)  # Both empty = perfect match
                        text_matches += 1
                    else:
                        similarity_scores.append(0.0)  # One empty, one not = no match

                valid_entries += 1

            except Exception as e:
                errors.append(f"Entry {idx}: Error in similarity calculation - {str(e)}")

        if errors:
            logger.warning(f"Similarity analysis warnings: {'; '.join(errors)}")

        if valid_entries == 0:
            raise ValueError("No valid entries found for similarity analysis.")

        # Calculate and return statistics
        return {
            "title_matches": title_matches,
            "title_match_percentage": (title_matches / valid_entries) * 100,
            "text_matches": text_matches,
            "text_match_percentage": (text_matches / valid_entries) * 100,
            "average_similarity_score": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0,
            "total_entries_analyzed": valid_entries,
            "entries_with_errors": len(errors),
            "similarity_threshold_used": similarity_threshold
        }


if __name__ == '__main__':
    # Example usage of the Evaluator class
    parser = argparse.ArgumentParser(description="Evaluate text transformation results.")

    parser.add_argument(
        "--results_path",
        type=str,
        default="./data/simple_transformed.json",
        help="Path to the JSON file containing evaluation results"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["length", "similarity"],
        default="length",
    )

    # Parse command line arguments
    args = parser.parse_args()

    evaluator = Evaluator(results_path=args.results_path)
    try:
        logger.info("Starting evaluation...")
        if args.metric == "length":
            logger.info("Evaluating length statistics...")
            length_stats = evaluator.length_eval()
            logger.info(f"Length Evaluation Results: {length_stats}")
        elif args.metric == "similarity":
            logger.info("Evaluating similarity statistics...")
            sim_stats = evaluator.similarity_eval()
            logger.info(f"Similarity Evaluation Results: {sim_stats}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
