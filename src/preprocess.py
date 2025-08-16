import random
import logging
import json
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass


from pathlib import Path

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data categories for the model."""
    SAFETY = "safety"
    DATA = "data"
    TEST = "test"


class ToneCategory(Enum):
    """Supported tones for the model."""
    DEFAULT = "default"
    DIRECT = "direct_language_version"
    SIMPLE = "simple_language_version"


@dataclass
class ContentEntry:
    """Structured entry for JSON content."""
    data_type: DataType
    tone: ToneCategory
    title: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        """Convert the entry to a dictionary."""
        return {
            "data_type": self.data_type.value,
            "tone": self.tone.value,
            "title": self.title,
            "text": self.text,
        }


class FewShotStrategySelector(Enum):
    """Selection strategy for few-shot examples."""
    BALANCED = "balanced"
    RANDOM = "random"


class DataPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data: dict = {}
        self.processed_data: Dict[DataType, Dict[ToneCategory, List[ContentEntry]]] = defaultdict(
            lambda: defaultdict(list))
        self.load_data()

    def load_data(self) -> None:
        """Extracts text from a JSON file."""
        try:
            path = Path(self.data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")

            with open(self.data_path, "r", encoding="utf-8") as file:
                self.raw_data: dict = json.load(file)

            logger.info(f"Data loaded successfully from {self.data_path}")
            self._process_raw_data()

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.data_path}: {e}")
            raise

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def _process_raw_data(self) -> None:
        """Processes raw data into a structured format."""
        self.processed_data.clear()
        for data, entries in self.raw_data.items():
            try:
                data_type = DataType(data)
            except ValueError:
                logger.warning(f"Unknown data type encountered: {data}")
                continue

            for tone, items in entries.items():
                try:
                    tone_category = ToneCategory(tone)
                except ValueError:
                    logger.warning(f"Unknown tone category encountered: {tone}")
                    continue

                for item in items:
                    entry = ContentEntry(
                        data_type=data_type,
                        tone=tone_category,
                        title=item["title"],
                        text=item["text"],
                    )
                    # Store the processed entries in the structured format
                    self.processed_data[data_type][tone_category].append(entry)

    def get_few_shot_examples(
            self,
            # data_type: Optional[DataType] = None,
            target_tone: Optional[ToneCategory] = None,
            num_samples: Optional[int] = None,
            shuffle: bool = True,
            strategy: Optional[FewShotStrategySelector] = None
    ) -> List[tuple[ContentEntry, Optional[ContentEntry]]]:
        """
        Get few-shot examples in a unified format.

        Args:
            target_tone (Optional[ToneCategory]): The tone category to transform to (e.g., SIMPLE, DIRECT).
            num_samples (Optional[int]): Number of samples to return. If None, returns all available examples.
            shuffle (bool): Whether to shuffle the returned examples.
            strategy (Optional[FewShotStrategySelector]): Strategy for selecting examples.

        Returns:
            List of tuples: (default_entry, transformed_entry or None)
        """
        all_examples = []
        selected_examples = []
        categories_to_check = [DataType.SAFETY, DataType.DATA]

        for cat in categories_to_check:
            if cat not in self.processed_data:
                continue

            default_entries = {e.title: e for e in self.processed_data[cat].get(ToneCategory.DEFAULT, [])}
            transformed_entries = {}
            if target_tone and target_tone != ToneCategory.DEFAULT:
                transformed_entries = {e.title: e for e in self.processed_data[cat].get(target_tone, [])}

            # Build tuples: (default, transformed or None)
            for title, default_entry in default_entries.items():
                transformed_entry = transformed_entries.get(title)
                all_examples.append((cat, default_entry, transformed_entry))

        # filter samples by strategy
        if strategy == FewShotStrategySelector.BALANCED:
            # Equal distribution across categories
            by_category = {}
            for cat in categories_to_check:
                by_category[cat] = [ex for ex in all_examples if ex[0] == cat]

            samples_per_cat = num_samples // len(categories_to_check)
            extra_samples = num_samples % len(categories_to_check)

            for i, cat in enumerate(categories_to_check):
                if not by_category[cat]:
                    continue

                cat_samples = samples_per_cat + (1 if i < extra_samples else 0)
                available = by_category[cat]
                selected_examples.extend(random.sample(available, min(len(available), cat_samples)))
        else:
            # fallback to random selection
            selected_examples = random.sample(all_examples, min(len(all_examples), num_samples))

        # Convert to final format (remove category info)
        result = [(ex[1], ex[2]) for ex in selected_examples]

        if shuffle:
            random.shuffle(result)
        return result

    def get_test_data(self) -> List[ContentEntry]:
        """
        Get test data entries for a specific tone.

        Returns:
            List of ContentEntry objects for the specified tone.
        """
        if DataType.TEST not in self.processed_data:
            logger.warning("No test data available.")
            return []

        test_entries = self.processed_data[DataType.TEST].get(ToneCategory.DEFAULT, [])
        return test_entries


if __name__ == '__main__':
    # Test the DataPreprocessor class
    data_path = "./data/fewshots.json"
    preprocessor = DataPreprocessor(data_path)
    try:
        content = preprocessor.get_few_shot_examples(
            target_tone=ToneCategory.DIRECT,
            num_samples=3,
            shuffle=False,
            strategy=FewShotStrategySelector.BALANCED
        )
        print(f"Retrieved {len(content)} content items:")
        for default_entry, transformed_entry in content:
            print("Title:", default_entry.title)
            print("Text:", default_entry.text[:50])
            if transformed_entry:
                print("Transformed Title:", transformed_entry.title)
                print("Transformed Text:", transformed_entry.text[:50])

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
