from .llm import chat, LLMError

SYSTEM = (
    "You are one role in a transparent multi-agent research workflow. "
    "Be concise, distinguish evidence from assumptions, never invent sources, "
    "and explicitly say when information is missing."
)

def _run_agent(name, task, context, mode):
    messages = [
        {"role": "system", "content": SYSTEM + f" Your role is: {name}."},
        {"role": "user", "content": f"TASK:\n{task}\n\nPROJECT CONTEXT:\n{context}"},
    ]
    answer, backend = chat(messages, mode=mode)
    return {"agent": name, "output": answer, "backend": backend}

def run_workflow(topic, audience, purpose, source_notes="", mode="Hybrid (API first, local fallback)"):
    context = (
        f"Topic: {topic}\nTarget audience: {audience}\nPurpose: {purpose}\n"
        f"User-provided notes and source material:\n{source_notes or '[No source notes supplied]'}"
    )
    outputs = []

    plan = _run_agent(
        "Planner Agent",
        "Create a focused research plan: 4-6 questions to investigate, scope boundaries, "
        "and what evidence would be useful. Do not answer the questions yet.",
        context, mode,
    )
    outputs.append(plan)

    research = _run_agent(
        "Research Synthesizer Agent",
        "Using ONLY the supplied project context and the research plan below, produce "
        "a structured evidence brief. If no source notes were supplied, clearly label "
        "claims as general background or items requiring external verification. "
        "Do not fabricate citations, URLs, statistics, or studies.\n\n"
        + plan["output"],
        context, mode,
    )
    outputs.append(research)

    analysis = _run_agent(
        "Analysis Agent",
        "Analyze the evidence brief. Identify key themes, implications, trade-offs, "
        "counterarguments, and unanswered questions. Separate supported observations "
        "from inference.\n\nEVIDENCE BRIEF:\n" + research["output"],
        context, mode,
    )
    outputs.append(analysis)

    review = _run_agent(
        "Evidence Review Agent",
        "Audit the draft below for unsupported claims, missing evidence, overconfident "
        "wording, internal inconsistencies, and questions needing verification. "
        "You cannot independently establish truth; do not claim that you have fact-checked "
        "against external sources. Return a concise issue list and suggested corrections.\n\n"
        "EVIDENCE BRIEF:\n" + research["output"] + "\n\nANALYSIS:\n" + analysis["output"],
        context, mode,
    )
    outputs.append(review)

    final = _run_agent(
        "Report Writer Agent",
        "Write a polished, well-structured research report for the stated audience and purpose. "
        "Include: title, executive summary, scope, key findings, analysis, limitations, "
        "open questions, and conclusion. Incorporate the evidence review. Never invent "
        "references or present unsupported claims as established facts.\n\n"
        "PLAN:\n" + plan["output"] + "\n\nEVIDENCE:\n" + research["output"]
        + "\n\nANALYSIS:\n" + analysis["output"] + "\n\nREVIEW:\n" + review["output"],
        context, mode,
    )
    outputs.append(final)

    return {"topic": topic, "audience": audience, "purpose": purpose,
            "agents": outputs, "report": final["output"]}

def demo_workflow(topic, audience, purpose, source_notes=""):
    """Transparent no-credentials fallback for UI testing; not an actual LLM run."""
    note = source_notes.strip() or "No source notes were supplied. External research is required."
    report = (
        f"# {topic}\n\n## Executive summary\n"
        f"This demonstration report outlines a research workflow for {topic}. "
        "It is a template-style demonstration, not an AI-generated evidence review.\n\n"
        f"## Audience and purpose\nAudience: {audience}\n\nPurpose: {purpose}\n\n"
        f"## Supplied material\n{note}\n\n"
        "## Findings to investigate\n"
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
    return {"topic": topic, "audience": audience, "purpose": purpose,
            "agents": [{"agent": n, "output": o, "backend": "DEMO — no LLM"} for n, o in stages],
            "report": report}
