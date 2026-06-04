import os
import time
import uuid
from dotenv import load_dotenv
import openai
import anthropic
from groq import Groq
from langfuse import Langfuse

load_dotenv()

# Initialize clients
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _log_to_langfuse(name, model, prompt, output, input_tokens, output_tokens, latency, provider):
    """Universal Langfuse v4 logger"""
    try:
        langfuse.create_event(
            name=name,
            input={"prompt": prompt},
            output={"response": output},
            metadata={
                "model": model,
                "provider": provider,
                "latency_seconds": round(latency, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        langfuse.flush()
    except Exception as e:
        print(f"  [Langfuse log skipped: {e}]")
        return str(uuid.uuid4())


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> dict:
    start = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    output = response.choices[0].message.content

    _log_to_langfuse(
        name="openai-call",
        model=model,
        prompt=prompt,
        output=output,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=latency,
        provider="openai"
    )

    return {
        "output": output,
        "model": model,
        "latency": latency,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }


def call_anthropic(prompt: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    start = time.time()
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    output = response.content[0].text

    _log_to_langfuse(
        name="anthropic-call",
        model=model,
        prompt=prompt,
        output=output,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency=latency,
        provider="anthropic"
    )

    return {
        "output": output,
        "model": model,
        "latency": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> dict:
    start = time.time()
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    output = response.choices[0].message.content

    _log_to_langfuse(
        name="groq-call",
        model=model,
        prompt=prompt,
        output=output,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency=latency,
        provider="groq"
    )

    return {
        "output": output,
        "model": model,
        "latency": latency,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }