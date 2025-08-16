import os
import json
import time
from openai import AzureOpenAI, APIError
from dataclasses import dataclass
from typing import List, Optional
import logging

from src.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """
    Structured response from the model containing content and metadata.
    """
    content: str
    model: Optional[str] = None
    finished: Optional[str] = None
    response_time: Optional[float] = None


@dataclass
class ModelClientError:
    """
    Error structure for model client operations.
    """
    message: str


class ModelClient:
    """
    Client for model calling and response generation using Azure OpenAI.
    """
    def __init__(self):
        """
        Initialize the ModelClient with configuration and Azure OpenAI client.
        """
        self.config = ModelConfig()
        self.client = AzureOpenAI(
            api_key=self.config.azure_api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.azure_endpoint,
        )

    def generate_response(
            self,
            messages: List[dict],
            max_completion_tokens: int = 512,
            temperature: float = 0.1,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0
    ):
        """
        Generate a response from the model based on the provided messages.

        Args:
            messages (List[dict]): List of messages to send to the model.
            max_completion_tokens (int): Maximum number of tokens for the completion.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Frequency penalty for token generation.
            presence_penalty (float): Presence penalty for token generation.

        Returns:
            ModelResponse: The content of the model's response along with metadata.
        """

        start_time = time.time()
        try:
            if not messages or not isinstance(messages, list):
                raise ValueError("Messages must be a non-empty list.")

            logger.info(f"Generating response with model {self.config.deployment}")
            logger.debug(f"Parameters: tokens={max_completion_tokens}, temp={temperature}, top_p={top_p}")

            response = self.client.chat.completions.create(
                model=self.config.deployment,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            elapsed_time = time.time() - start_time

            if not response.choices or len(response.choices) == 0:
                raise ValueError("No choices returned in the response.")

            model_response = ModelResponse(
                content=response.choices[0].message.content,
                model=self.config.deployment,
                finished=response.choices[0].finish_reason,
                response_time=elapsed_time
            )

            return model_response

        except APIError as api_error:
            logging.error(f"API error occurred: {api_error}")
            return ModelClientError(
                message=f"API error: {api_error.message}"
            )

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return ModelClientError(
                message=str(e)
            )
