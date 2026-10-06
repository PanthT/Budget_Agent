"""Budget Agent configuration.

Reads model/provider settings from .env and returns a LangChain chat model.
Defaults to local Ollama (free); switch to OpenRouter via LLM_PROVIDER=openrouter.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def get_chat_model():
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    # Default: local Ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.getenv("LLM_MODEL_NAME", "llama3.2:3b"))