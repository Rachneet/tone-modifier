# Generic system prompt that works for any tone
GENERIC_SYSTEM_PROMPT = """You are an expert content writer and tone specialist. Your job is to rewrite text in the specified tone while preserving all important information."""

# Specific tone prompts
SIMPLE_TONE_PROMPT = """Your task is to rewrite ONLY the text content using simple, clear language. Follow these rules:

1. Keep the title EXACTLY the same - do not change it
2. Simplify the text so it is understandable by a five year old
3. Keep all important information - don't remove key details
4. The length of the text should be around {answer_length} words

Format your response as a JSON object with the following structure:
{{
    "Title": "[keep original title unchanged]",
    "Text": "[text in simple tone here]"
}}"""

DIRECT_TONE_PROMPT = """Your task is to rewrite ONLY the text content using direct, action-oriented language. Follow these rules:

1. Keep the title EXACTLY the same - do not change it
2. Rewrite the text to be direct and commanding tone
3. Keep all important information - don't remove key details
4. The length of the text should be around {answer_length} words

Format your response as a JSON object with the following structure:
{{
    "Title": "[keep original title unchanged]",
    "Text": "[text in direct tone here]"
}}"""
