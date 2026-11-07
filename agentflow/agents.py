from .llm import chat
import time

SYSTEM = (
    "You are one role in a transparent multi-agent research workflow. "
    "Be concise, distinguish source-supported statements from inference, "
    "never invent sources or statistics, and identify missing information."
)

# These output budgets apply to Ollama only. The API request path ignores them.
TOKEN_BUDGETS = {
    "Planner Agent": 400,
    "Research Synthesizer Agent": 600,
    "Analysis Agent": 550,
    "Evidence Review Agent": 450,
    "Report Writer Agent": 900,
}


def _run_agent(name, task, context, mode):
    messages = [
        {"role": "system", "content": SYSTEM + f" Your role is: {name}."},
        {"role": "user", "content": f"TASK:\n{task}\n\nPROJECT CONTEXT:\n{context}"},
    ]
    started = time.perf_counter()
    answer, backend = chat(messages, mode=mode, max_tokens=TOKEN_BUDGETS.get(name, 500))
    elapsed = time.perf_counter() - started
    return {"agent": name, "output": answer, "backend": backend,
            "elapsed_seconds": round(elapsed, 2)}


def run_workflow(topic, audience, purpose, source_notes="",
                 mode="Hybrid (API first, local fallback)"):
    context = (
        f"Topic: {topic}\nTarget audience: {audience}\nPurpose: {purpose}\n"
        f"User-provided notes and source material:\n"
        f"{source_notes or '[No source notes supplied]'}"
    )
    outputs = []

    plan = _run_agent(
        "Planner Agent",
        "Create a concise research plan: 4-5 questions, scope boundaries, and useful "
        "evidence types. Do not answer the questions.",
        context, mode,
    )
    outputs.append(plan)

    research = _run_agent(
        "Research Synthesizer Agent",
        "Using ONLY the project context and plan, create a concise evidence brief mapped "
        "to the research questions. Separate supplied evidence from claims needing "
        "external verification. If notes are absent, label claims as general background. "
        "Never invent citations, URLs, statistics, or studies.\n\nPLAN:\n" + plan["output"],
        context, mode,
    )
    outputs.append(research)

    analysis = _run_agent(
        "Analysis Agent",
        "Analyze the evidence brief. Identify supported observations, labelled inferences, "
        "themes, trade-offs, counterarguments, and unanswered questions. Do not introduce "
        "unsupported facts.\n\nEVIDENCE BRIEF:\n" + research["output"],
        context, mode,
    )
    outputs.append(analysis)

    review = _run_agent(
        "Evidence Review Agent",
        "Audit the evidence brief and analysis for unsupported claims, missing evidence, "
        "overconfident wording, inconsistencies, and questions needing verification. You "
        "cannot independently verify truth. Return at most 5 numbered findings. For each "
        "include severity (High/Medium/Low), claim/section, why problematic, required "
        "correction, and whether supplied material can resolve it. Do not invent sources."
        "\n\nEVIDENCE BRIEF:\n" + research["output"] +
        "\n\nANALYSIS:\n" + analysis["output"],
        context, mode,
    )
    outputs.append(review)

    final = _run_agent(
        "Report Writer Agent",
        "Write a concise, polished report for the stated audience and purpose. Treat every "
        "review finding as mandatory: correct, qualify, remove, or retain as unresolved. "
        "Distinguish supported claims, inferences, and external context. Never claim "
        "external fact-checking or invent references. Include title, executive summary, "
        "scope, key findings, analysis, limitations, open questions, an 'Evidence Review "
        "Response' mapping each finding to its action, and conclusion.\n\nEVIDENCE BRIEF:\n"
        + research["output"] + "\n\nANALYSIS:\n" + analysis["output"]
        + "\n\nEVIDENCE REVIEW:\n" + review["output"],
        context, mode,
    )
    outputs.append(final)

    return {
        "topic": topic, "audience": audience, "purpose": purpose,
        "agents": outputs, "report": final["output"], "review": review["output"],
        "total_agent_seconds": round(sum(a.get("elapsed_seconds", 0) for a in outputs), 2),
    }


def demo_workflow(topic, audience, purpose, source_notes=""):
    """Transparent no-credentials fallback for UI testing; not an actual LLM run."""
    note = source_notes.strip() or "No source notes were supplied. External research is required."
    report = (
        f"# {topic}\n\n## Executive summary\n"
        f"This demonstration report outlines a research workflow for {topic}. "
        "It is a template-style demonstration, not an AI-generated evidence review.\n\n"
        f"## Audience and purpose\nAudience: {audience}\n\nPurpose: {purpose}\n\n"
        f"## Supplied material\n{note}\n\n## Findings to investigate\n"
        "- Define the central problem and its context.\n"
        "- Gather reliable, dated primary and secondary sources.\n"
        "- Compare evidence, limitations, and competing explanations.\n\n"
        "## Limitations\nNo language model was connected. No external sources were checked. "
        "Do not submit this demonstration output as completed research.\n\n"
        "## Conclusion\nConnect conclusions to verified evidence before using this report academically."
    )
    stages = [
        ("Planner Agent", "Demo plan: define scope, questions, evidence needs, and boundaries."),
        ("Research Synthesizer Agent", f"Demo evidence input:\n{note}"),
        ("Analysis Agent", "Demo analysis: organize themes and identify trade-offs after evidence collection."),
        ("Evidence Review Agent", "Demo review: verify all factual claims against reliable external sources."),
        ("Report Writer Agent", report),
    ]
    return {
        "topic": topic, "audience": audience, "purpose": purpose,
        "agents": [{"agent": n, "output": o, "backend": "DEMO — no LLM", "elapsed_seconds": 0.0}
                   for n, o in stages],
        "report": report, "review": stages[3][1], "total_agent_seconds": 0.0,
    }