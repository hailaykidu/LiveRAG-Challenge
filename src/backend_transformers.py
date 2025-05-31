import json
import re
from typing import Tuple
from loguru import logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

    device_type = next(iter(model.parameters())).device.type
    device = torch.device(device_type)
    if device_type != "cuda":
        logger.warning("No GPU available. Using CPU for LLM!")

    return tokenizer, model, device


def chat_completions(
    initialized_model: Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device],
    prompt: str,
    model: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.1,  # High for creativity, low for technical/precise answer
    format: str = "text",
) -> str | dict:
    tokenizer, model, device = initialized_model

    if format == "json_object":
        system_prompt += " Please return the answer strictly in valid JSON format, without additional explanations."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    with torch.inference_mode():
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs,
            temperature=temperature,
            max_length=31000,  # max_number of token (input inclusive)
            # top_p=0.9, # High for wider range of tokens, low for restricting to most likely token
            # Force correct eos_, pad_token usage
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_length = inputs.shape[1]
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

        if format == "json_object":
            return response_to_json(response)
        else:
            return response


def response_to_json(response: str) -> dict:
    try:
        if response.startswith("```"):
            code_block_match = re.search(r"^```(?:json)?(.+?)```", response, re.DOTALL)
            if code_block_match:
                response = code_block_match.group(1).strip()
                logger.trace(f"Extracted JSON from code block: {response}")

        response = json.loads(response)
        logger.trace(f"Parsed JSON response: {response}")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON object in response: {response}")
    return response
