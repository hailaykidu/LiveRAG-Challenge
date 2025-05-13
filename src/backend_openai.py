from openai import OpenAI
from loguru import logger
import os


def load_model(model_name: str) -> OpenAI:
    openai = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
        api_key=os.environ.get("OPENAI_API_KEY", None),
    )
    return openai


def chat_completions(
    openai: OpenAI,
    prompt: str,
    model: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.1,
    format: str = "text",
    retries: int = 5,
    fail_after_retries: bool = False,
    response_reties_exceeded: str = "I apologise, I can't help you there.",
) -> str:
    for attempt in range(retries):  # this retries is only protecting us from an empty answer
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": format},
        )

        answer = completion.choices[0].message.content
        if answer:
            return answer

        logger.debug(f"Empty answer received on attempt {attempt + 1}/{retries}, retrying...")

    if fail_after_retries:
        raise Exception("Could not generate a valid response after maximum retries.")

    logger.warning("Could not generate completion. Continue with default response.")
    return response_reties_exceeded
