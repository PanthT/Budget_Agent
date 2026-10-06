"""Budget Agent configuration.

Loads local Ollama settings and disables external LangSmith tracing.
"""
import ipaddress
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _get_local_ollama_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        _ = parsed.port
        is_loopback = hostname == "localhost"
        if hostname is not None and not is_loopback:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = False
        parsed = None

    if (
        parsed is None
        or parsed.scheme != "http"
        or not is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "OLLAMA_BASE_URL must be a local HTTP URL such as "
            "http://localhost:11434; remote endpoints are not allowed."
        )

    return base_url


def get_chat_model():
    from langchain_ollama import ChatOllama

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    if "cloud" in model_name.lower():
        raise ValueError("Cloud-hosted Ollama models are not allowed for local-only data.")

    return ChatOllama(
        model=model_name,
        base_url=_get_local_ollama_url(),
        sync_client_kwargs={"trust_env": False},
        async_client_kwargs={"trust_env": False},
    )
