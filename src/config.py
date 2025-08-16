import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    model_name: str = "gpt-4.1"
    azure_api_key: str = os.getenv("AZURE_API_KEY", "")
    azure_endpoint: str = "https://famai-open-ai.openai.azure.com/"
    deployment: str = "gpt-41_interviews_2508"
    api_version: str = "2024-12-01-preview"
