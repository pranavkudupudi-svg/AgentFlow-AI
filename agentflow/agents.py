from .llm import chat
import time


SYSTEM = (
    "You are one role in a transparent multi-agent research workflow. Be concise "
    "and structured. Never invent sources, citations, authors, titles, journals, "
    "dates, statistics, institutions, case studies, or geographic scope. "
    "Distinguish supplied evidence from inference. If information is absent, say so."
)


# Output budgets for API-based generation.
API_TOKEN_BUDGETS = {
    "Planner Agent": 500,
    "Research Synthesizer Agent": 1000,
    "Analysis Agent": 900,
    "Evidence Review Agent": 1100,
    "Report Writer Agent": 1800,
}


# Smaller budgets for CPU-bound local inference.
LOCAL_TOKEN_BUDGETS = {
    "Planner Agent": 250,
    "Research Synthesizer Agent": 450,
    "Analysis Agent": 400,
    "Evidence Review Agent": 500,
    "Report Writer Agent": 700,
}


def _run_agent(name, task, context, mode):
    messages = [
        {
            "role": "system",
            "content": SYSTEM + f" Your role is: {name}."
        },
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"PROJECT CONTEXT:\n{context}"
            )
        },
    ]

    started = time.perf_counter()

    budgets = (
        LOCAL_TOKEN_BUDGETS
        if mode == "Local only"
        else API_TOKEN_BUDGETS
    )

    budget = budgets.get(name, 800)

    answer, backend = chat(
        messages,
        mode=mode,
        max_tokens=budget,
    )

    return {
        "agent": name,
        "output": answer,
        "backend": backend,
        "elapsed_seconds": round(
            time.perf_counter() - started, 2
        ),
    }


def run_workflow(
    topic,
    audience,
    purpose,
    source_notes="",
    mode="Hybrid (API first, local fallback)",
):
    notes = source_notes.strip()

    source_context = notes or (
        "[NO SOURCE NOTES PROVIDED. Do not create or name studies, "
        "publications, surveys, case studies, institutions, authors, "
        "dates, or statistics. State that evidence collection is required.]"
    )

    metadata_context = (
        f"Topic: {topic}\n"
        f"Target audience: {audience}\n"
        f"Purpose: {purpose}\n"
        "Geographic scope: Not specified by user; do not assume a region."
    )

    context = (
        metadata_context
        + "\nUSER-PROVIDED SOURCE MATERIAL "
        + "(only permitted source inventory):\n"
        + source_context
    )

    outputs = []

    # 1. Planner Agent
    plan = _run_agent(
        "Planner Agent",
        (
            "Create 3-4 research questions, scope boundaries, and useful "
            "evidence types. Do not answer the questions or invent a "
            "geographic region, institution, or time period. "
            "If scope is unspecified, say so."
        ),
        metadata_context,
        mode,
    )
    outputs.append(plan)

    # 2. Research Synthesizer Agent
    research = _run_agent(
        "Research Synthesizer Agent",
        (
            "Create a concise evidence inventory using ONLY the "
            "user-provided source material. Repeat bibliographic details "
            "only if explicitly present. Never fill missing titles, authors, "
            "journals, years, institutions, URLs, or findings from memory. "
            "Do not turn suggested evidence types into sources already found. "
            "If no source notes were supplied, say no source evidence was "
            "provided and list evidence still needed. Separate supplied "
            "evidence, unverified user claims, and evidence gaps.\n\n"
            "PLAN:\n" + plan["output"]
        ),
        context,
        mode,
    )
    outputs.append(research)

    # 3. Analysis Agent
    analysis = _run_agent(
        "Analysis Agent",
        (
            "Analyze only the evidence inventory. Add no factual examples, "
            "studies, statistics, institutions, or claims absent from it. "
            "If no actual sources exist, provide an issue framework and "
            "state that substantive conclusions cannot be established. "
            "Separate supported observations, inferences, and open questions."
            "\n\nEVIDENCE INVENTORY:\n"
            + research["output"]
        ),
        metadata_context,
        mode,
    )
    outputs.append(analysis)

    # 4. Evidence Review Agent
    review = _run_agent(
        "Evidence Review Agent",
        (
            "Strictly audit every named source, author, publication, "
            "institution, date, statistic, case study, and factual claim "
            "in the evidence inventory and analysis against the "
            "USER-PROVIDED SOURCE MATERIAL in project context. "
            "Do not treat another agent's assertion as proof. "
            "Flag anything not explicitly present as UNVERIFIED/UNSUPPORTED. "
            "Also flag invented geographic scope and unsupported causal "
            "claims. Return at most 5 prioritized findings. For each include "
            "severity, exact claim/detail, traceability status, why it fails, "
            "and required correction. If no source notes were supplied, say "
            "source-level verification is impossible. Do not invent sources."
            "\n\nEVIDENCE INVENTORY:\n"
            + research["output"]
            + "\n\nANALYSIS:\n"
            + analysis["output"]
        ),
        context,
        mode,
    )
    outputs.append(review)

    # 5. Report Writer Agent
    final = _run_agent(
        "Report Writer Agent",
        (
            "Write a concise report without implying external research was "
            "conducted. Use only traceable information from user-provided "
            "source material. Do not name or cite a source unless its details "
            "appear in the original notes. Never convert source suggestions "
            "or unsupported statements into findings. If no source notes were "
            "supplied, title the output 'Preliminary Research Framework' and "
            "explain evidence collection is required. Address every reviewer "
            "finding by correcting, qualifying, removing, or marking it "
            "unresolved. Include title, executive summary, scope, source "
            "basis, supported observations (if any), analysis/inferences, "
            "limitations, open questions, Evidence Review Response, and "
            "conclusion.\n\n"
            "EVIDENCE INVENTORY:\n"
            + research["output"]
            + "\n\nANALYSIS:\n"
            + analysis["output"]
            + "\n\nREVIEW FINDINGS:\n"
            + review["output"]
        ),
        context,
        mode,
    )
    outputs.append(final)

    return {
        "topic": topic,
        "audience": audience,
        "purpose": purpose,
        "agents": outputs,
        "report": final["output"],
        "review": review["output"],
        "total_agent_seconds": round(
            sum(a.get("elapsed_seconds", 0) for a in outputs),
            2,
        ),
    }


def demo_workflow(topic, audience, purpose, source_notes=""):
    """Transparent no-credentials fallback for UI testing; not an actual LLM run."""

    note = (
        source_notes.strip()
        or "No source notes were supplied. External research is required."
    )

    report = (
        f"# {topic}\n\n"
        "## Executive summary\n"
        f"This demonstration report outlines a research workflow for {topic}. "
        "It is a template-style demonstration, not an AI-generated evidence review.\n\n"
        "## Audience and purpose\n"
        f"Audience: {audience}\n\n"
        f"Purpose: {purpose}\n\n"
        "## Supplied material\n"
        f"{note}\n\n"
        "## Findings to investigate\n"
        "- Define the central problem and its context.\n"
        "- Gather reliable, dated primary and secondary sources.\n"
        "- Compare evidence, limitations, and competing explanations.\n\n"
        "## Limitations\n"
        "No language model was connected. No external sources were checked. "
        "Do not submit this demonstration output as completed research.\n\n"
        "## Conclusion\n"
        "Connect conclusions to verified evidence before using this report academically."
    )

    stages = [
        (
            "Planner Agent",
            "Demo plan: define scope, questions, evidence needs, and boundaries."
        ),
        (
            "Research Synthesizer Agent",
            f"Demo evidence input:\n{note}"
        ),
        (
            "Analysis Agent",
            "Demo analysis: organize themes and identify trade-offs after evidence collection."
        ),
        (
            "Evidence Review Agent",
            "Demo review: verify all factual claims against reliable external sources."
        ),
        (
            "Report Writer Agent",
            report
        ),
    ]

    return {
        "topic": topic,
        "audience": audience,
        "purpose": purpose,
        "agents": [
            {
                "agent": name,
                "output": output,
                "backend": "DEMO — no LLM",
                "elapsed_seconds": 0.0,
            }
            for name, output in stages
        ],
        "report": report,
        "review": stages[3][1],
        "total_agent_seconds": 0.0,
    }
