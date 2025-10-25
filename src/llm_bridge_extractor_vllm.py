#!/usr/bin/env python3
"""
LLM Bridge Extractor using vLLM API (OpenAI-compatible)

Uses existing vLLM server instead of loading model directly.
Keeps same interface as llm_bridge_extractor.py for easy drop-in replacement.
"""

import requests
import logging
from typing import Tuple

# Import prompt templates (same as original)
from prompt_templates import (
    build_pregeneration_prompt,
    build_final_answer_prompt,
    parse_llm_response
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# vLLM server configuration
VLLM_API_BASE = "http://localhost:8013/v1"
VLLM_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
VLLM_API_KEY = "sk-local"  # vLLM API key from server config


def load_model(model_name: str = "Qwen/Qwen2.5-32B-Instruct", device_map: str = "auto"):
    """
    Dummy load_model for compatibility with existing code.

    vLLM server is already running, so we don't actually load anything.
    Just return None, None to signal vLLM mode.

    Args:
        model_name: Model name (ignored, uses vLLM server's model)
        device_map: Device mapping (ignored)

    Returns:
        (None, None) to signal vLLM API mode
    """
    logger.info(f"Using vLLM API at {VLLM_API_BASE}")
    logger.info(f"Model: {VLLM_MODEL_NAME}")

    # Test connection
    try:
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        response = requests.get(f"{VLLM_API_BASE}/models", headers=headers, timeout=5)
        if response.status_code == 200:
            logger.info("✅ vLLM server connection successful!")
        else:
            logger.warning(f"⚠️  vLLM server responded with status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to vLLM server: {e}")
        raise ConnectionError(f"Cannot connect to vLLM server at {VLLM_API_BASE}")

    return None, None


def call_llm_with_prompt(
    prompt: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 400,
    temperature: float = 0.3
) -> Tuple[str, int, int]:
    """
    Call vLLM API with a prompt and return response with token counts.

    Args:
        prompt: Complete prompt string
        model: Ignored (for compatibility)
        tokenizer: Ignored (for compatibility)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    # Prepare API request
    api_url = f"{VLLM_API_BASE}/chat/completions"

    # Format as chat messages
    messages = [
        {"role": "system", "content": "You are a Knowledge Graph Question Answering assistant."},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }

    # Call vLLM API
    try:
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()

        # Extract response and token counts
        response_text = result['choices'][0]['message']['content']
        input_tokens = result['usage']['prompt_tokens']
        output_tokens = result['usage']['completion_tokens']

        logger.info(f"vLLM call: {input_tokens} input tokens, {output_tokens} output tokens")

        return response_text, input_tokens, output_tokens

    except requests.exceptions.Timeout:
        logger.error("vLLM API call timed out (120s)")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"vLLM API call failed: {e}")
        raise
    except KeyError as e:
        logger.error(f"Unexpected vLLM API response format: {e}")
        logger.error(f"Response: {result}")
        raise


def format_triplets_for_prompt(triplets, limit: int = 30) -> str:
    """
    Format triplets for the prompt in indexed format.
    Filter out metadata triplets (type/common/freebase).

    (Copied from original for compatibility)
    """
    formatted = []
    idx = 1

    for t in triplets[:limit * 2]:  # Check more to get enough after filtering
        if len(t) < 3:
            continue

        # Skip metadata predicates
        pred = str(t[1])
        if any(pred.startswith(p) for p in ['type.', 'common.', 'freebase.']):
            continue

        # Format entities - use plain text
        subject = str(t[0])
        obj = str(t[2])

        formatted.append(f"[{idx}] {subject} --{pred}--> {obj}")
        idx += 1

        if len(formatted) >= limit:
            break

    return "\n".join(formatted)


if __name__ == "__main__":
    # Test vLLM connection
    print("="*80)
    print("Testing vLLM API Connection")
    print("="*80)

    try:
        model, tokenizer = load_model()
        print("\n✅ vLLM server is ready!")

        # Test a simple prompt
        test_prompt = "What is 2 + 2?"
        print(f"\nTest prompt: {test_prompt}")

        response, input_tok, output_tok = call_llm_with_prompt(
            test_prompt,
            max_new_tokens=50,
            temperature=0.1
        )

        print(f"\nResponse: {response}")
        print(f"Tokens: {input_tok} in, {output_tok} out")
        print("\n✅ vLLM API working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure vLLM server is running on port 8013:")
        print("  vllm serve Qwen/Qwen2.5-32B-Instruct --port 8013")
