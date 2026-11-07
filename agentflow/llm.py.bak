import os
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMError(RuntimeError):
    pass

def _api_chat(messages, temperature=0.3):
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise LLMError("LLM_API_KEY is not configured.")
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as exc:
        raise LLMError(f"API request failed: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Unexpected API response: {exc}") from exc

def _ollama_chat(messages, temperature=0.3):
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    try:
        response = requests.post(
            f"{base}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise LLMError(f"Unexpected Ollama response: {exc}") from exc

def chat(messages, mode="Hybrid (API first, local fallback)", temperature=0.3):
    """Hybrid: try configured API first, then Ollama; local/API-only modes are also supported."""
    errors = []
    if mode == "API only":
        return _api_chat(messages, temperature), "API"
    if mode == "Local only":
        return _ollama_chat(messages, temperature), "Ollama"
    for name, fn in (("API", _api_chat), ("Ollama", _ollama_chat)):
        try:
            return fn(messages, temperature), name
        except LLMError as exc:
            errors.append(f"{name}: {exc}")
    raise LLMError("No LLM backend available. " + " | ".join(errors))
