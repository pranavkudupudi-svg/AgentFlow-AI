
import os
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    pass


class LLMTruncationError(LLMError):
    """Raised when generation remains incomplete after continuation."""

    def __init__(self, message, partial_output=""):
        super().__init__(message)
        self.partial_output = partial_output


MAX_CONTINUATIONS = 1


def _api_chat(messages, temperature=0.3, max_tokens=1000):
    key = os.getenv("LLM_API_KEY", "").strip()

    if not key:
        raise LLMError("LLM_API_KEY is not configured.")

    base = os.getenv(
        "LLM_BASE_URL",
        "https://api.openai.com/v1"
    ).rstrip("/")

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason")

        if not content.strip():
            raise LLMError("API returned an empty response.")

        if finish_reason == "length":
            raise LLMTruncationError(
                f"API reached its {max_tokens}-token output limit.",
                partial_output=content.strip(),
            )

        return content.strip()

    except LLMTruncationError:
        raise
    except requests.RequestException as exc:
        raise LLMError(f"API request failed: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Unexpected API response: {exc}") from exc


def _ollama_chat(messages, temperature=0.3, max_tokens=1000):
    base = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    ).rstrip("/")

    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    try:
        response = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=600,
        )

        response.raise_for_status()
        data = response.json()

        content = data["message"]["content"] or ""
        done_reason = data.get("done_reason")

        if not content.strip():
            raise LLMError("Ollama returned an empty response.")

        if done_reason == "length":
            raise LLMTruncationError(
                f"Ollama reached its {max_tokens}-token output limit.",
                partial_output=content.strip(),
            )

        return content.strip()

    except LLMTruncationError:
        raise
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise LLMError(f"Unexpected Ollama response: {exc}") from exc


def _continue_output(
    original_messages,
    partial_output,
    backend_function,
    temperature,
    max_tokens,
    backend_name,
):
    """Continue a truncated response once using the same backend."""

    continuation_messages = list(original_messages)

    continuation_messages.append({
        "role": "assistant",
        "content": partial_output,
    })

    continuation_messages.append({
        "role": "user",
        "content": (
            "Continue the previous response from exactly where it stopped. "
            "Do not repeat any earlier content, headings, or sentences. "
            "Complete the remaining response and preserve its original format."
        ),
    })

    try:
        continuation = backend_function(
            continuation_messages,
            temperature,
            max_tokens,
        )

        return partial_output.rstrip() + "\n" + continuation.lstrip()

    except LLMTruncationError as exc:
        combined_output = (
            partial_output.rstrip()
            + "\n"
            + exc.partial_output.lstrip()
        )

        raise LLMTruncationError(
            f"{backend_name} response was still truncated after "
            f"{MAX_CONTINUATIONS} continuation attempt(s).",
            partial_output=combined_output,
        ) from exc


def _run_with_continuation(
    messages,
    backend_function,
    backend_name,
    temperature,
    max_tokens,
):
    try:
        return backend_function(
            messages,
            temperature,
            max_tokens,
        )

    except LLMTruncationError as exc:
        if not exc.partial_output:
            raise

        return _continue_output(
            messages,
            exc.partial_output,
            backend_function,
            temperature,
            max_tokens,
            backend_name,
        )


def chat(
    messages,
    mode="Hybrid (API first, local fallback)",
    temperature=0.3,
    max_tokens=1000,
):
    """
    API only: use the configured API.
    Local only: use Ollama.
    Hybrid: try API first, then Ollama if the API request fails.

    Provider-reported truncation triggers one continuation attempt
    using the same backend.
    """

    if mode == "API only":
        return _run_with_continuation(
            messages,
            _api_chat,
            "API",
            temperature,
            max_tokens,
        ), "API"

    if mode == "Local only":
        return _run_with_continuation(
            messages,
            _ollama_chat,
            "Ollama",
            temperature,
            max_tokens,
        ), "Ollama"

    try:
        return _run_with_continuation(
            messages,
            _api_chat,
            "API",
            temperature,
            max_tokens,
        ), "API"

    except LLMTruncationError:
        raise

    except LLMError as api_error:
        try:
            return _run_with_continuation(
                messages,
                _ollama_chat,
                "Ollama",
                temperature,
                max_tokens,
            ), "Ollama"

        except LLMError as ollama_error:
            raise LLMError(
                "No LLM backend available. "
                f"API: {api_error} | Ollama: {ollama_error}"
            ) from ollama_error
