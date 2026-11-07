# AgentFlow AI

A Streamlit-based multi-agent research and report generation prototype.

## Features
- Five sequential specialist roles: Planner, Research Synthesizer, Analyst, Evidence Reviewer, Report Writer.
- Hybrid LLM deployment: OpenAI-compatible API first, Ollama fallback.
- Local no-credential demo mode for interface testing.
- PDF and Markdown report export.
- Explicit evidence limitations: the reviewer flags unsupported claims but does not independently verify facts.

## 1. Requirements
- Python 3.10+
- Internet for API calls, or Ollama running locally for local inference.

## 2. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and configure either an OpenAI-compatible API or Ollama.

### API configuration
Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
For OpenAI-compatible providers, `LLM_BASE_URL` should point to the provider's `/v1` API root.

### Local Ollama configuration
Install Ollama, then run:

```bash
ollama pull llama3.2:3b
ollama serve
```

Set `OLLAMA_MODEL=llama3.2:3b` in `.env`.

## 3. Run

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (normally http://localhost:8501).

## 4. Test

```bash
pytest -q
```

Use the **Preview demo without an LLM** button to test the interface without credentials. Demo output is clearly labelled and is not a real model run.

## Architecture
User brief + optional source notes -> Planner -> Research Synthesizer -> Analyst -> Evidence Reviewer -> Report Writer -> Streamlit results + PDF/Markdown export.

## Academic note
This is a sequential multi-agent orchestration prototype. The agents are role-specialized LLM calls, not independently trained models. The evidence reviewer performs a consistency and support audit against provided context; it does not establish factual truth without external sources. Similarity of outputs does not constitute independent verification.

## Suggested evaluation
Test with 3 topics and compare:
- whether the plan covers the topic,
- whether the evidence brief separates supplied evidence from background,
- whether the analyst identifies trade-offs,
- whether the reviewer catches unsupported claims,
- whether the final report follows the requested structure.

Record actual observations and timings in your report; do not invent accuracy scores.
