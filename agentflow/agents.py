
import json
import time

from .llm import chat
from .evidence import (
    parse_evidence_output,
    apply_support_assessments,
    evidence_summary,
)


SYSTEM = (
    "You are one role in a transparent multi-agent research workflow. "
    "Be concise and structured. Never invent sources, citations, authors, "
    "titles, journals, dates, statistics, institutions, case studies, "
    "or geographic scope. Distinguish supplied evidence from inference. "
    "If information is absent, say so. Treat user-provided notes as "
    "unverified material, not automatically as established facts."
)


API_TOKEN_BUDGETS = {
    "Planner Agent": 500,
    "Research Synthesizer Agent": 1000,
    "Analysis Agent": 900,
    "Evidence Review Agent": 1100,
    "Report Writer Agent": 1800,
}


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
            "content": SYSTEM + f" Your role is: {name}.",
        },
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"PROJECT CONTEXT:\n{context}"
            ),
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


def _rule_based_agent(name, output):
    return {
        "agent": name,
        "output": output,
        "backend": "RULE-BASED — no source evidence",
        "elapsed_seconds": 0.0,
    }


def _run_without_sources(topic, audience, purpose, plan):
    research_output = (
        "# Evidence Inventory\n\n"
        "**Status: No source evidence supplied.**\n\n"
        "No articles, publications, datasets, reports, case studies, "
        "or other source materials were provided by the user.\n\n"
        "## Evidence still required\n"
        "- Relevant peer-reviewed research and publications.\n"
        "- Reliable datasets or official statistics, where applicable.\n"
        "- Documented case studies or real-world examples.\n"
        "- Source dates, authors, publication details, and links.\n\n"
        "No research findings can be established from the current input."
    )

    analysis_output = (
        "# Preliminary Issue Framework\n\n"
        f"**Topic:** {topic}\n\n"
        "No source evidence is available, so substantive factual "
        "conclusions cannot be established.\n\n"
        "## Areas to investigate\n"
        "- Define the central problem and its context.\n"
        "- Identify potential benefits and limitations.\n"
        "- Examine relevant stakeholders and their perspectives.\n"
        "- Identify risks, constraints, and unanswered questions.\n\n"
        "These are proposed investigation areas, not verified findings."
    )

    review_output = (
        "# Evidence Review\n\n"
        "**Overall status: INCOMPLETE — NO SOURCE MATERIAL.**\n\n"
        "No source-level verification is possible because no source "
        "notes were supplied. No substantive factual conclusion "
        "can be validated.\n\n"
        "**Required correction:** Collect source material and repeat "
        "the evidence synthesis and review stages."
    )

    report_output = (
        "# Preliminary Research Framework\n\n"
        "## Executive Summary\n"
        f"This document establishes a preliminary research framework "
        f"for **{topic}**. It is intended for **{audience}** and aims "
        f"to: {purpose}.\n\n"
        "No source material was supplied. Therefore, this document "
        "does not present completed research or verified findings.\n\n"
        "## Scope\n"
        f"The proposed research topic is {topic}. The geographic and "
        "temporal scope has not been specified.\n\n"
        "## Source Basis\n"
        "No research sources were supplied for this run.\n\n"
        "## Supported Observations\n"
        "No evidence-based observations can be established.\n\n"
        "## Analysis and Proposed Investigation\n"
        "- Define the research problem and key questions.\n"
        "- Identify reliable primary and secondary sources.\n"
        "- Compare findings, limitations, and competing explanations.\n\n"
        "These are proposed research activities, not factual conclusions.\n\n"
        "## Limitations\n"
        "- No source material was provided.\n"
        "- No external research or independent fact-checking was performed.\n"
        "- No empirical findings or statistical conclusions can be reported.\n\n"
        "## Evidence Review Response\n"
        "The evidence review is incomplete because source-level "
        "verification was not possible.\n\n"
        "## Open Questions\n"
        "- What geographic and temporal scope should be examined?\n"
        "- Which sources and datasets are relevant?\n"
        "- What criteria should be used to evaluate the evidence?\n\n"
        "## Conclusion\n"
        "This document is a preliminary research framework only. "
        "Collect and verify relevant evidence before drawing conclusions."
    )

    return [
        _rule_based_agent("Research Synthesizer Agent", research_output),
        _rule_based_agent("Analysis Agent", analysis_output),
        _rule_based_agent("Evidence Review Agent", review_output),
        _rule_based_agent("Report Writer Agent", report_output),
    ]


def run_workflow(
    topic,
    audience,
    purpose,
    source_notes="",
    mode="Hybrid (API first, local fallback)",
):
    notes = source_notes.strip()

    metadata_context = (
        f"Topic: {topic}\n"
        f"Target audience: {audience}\n"
        f"Purpose: {purpose}\n"
        "Geographic scope: Not specified by user; do not assume a region."
    )

    source_context = notes or (
        "[NO SOURCE NOTES PROVIDED. No factual research findings "
        "can be established.]"
    )

    context = (
        metadata_context
        + "\n\nUSER-PROVIDED SOURCE MATERIAL:\n"
        + source_context
    )

    outputs = []
    evidence_claims = []
    validation_error = None
    support_assessment_error = None

    # 1. Planner Agent
    plan = _run_agent(
        "Planner Agent",
        (
            "Create 3-4 research questions, scope boundaries, and "
            "useful evidence types. Do not answer the questions or "
            "invent a geographic region, institution, or time period. "
            "If scope is unspecified, say so."
        ),
        metadata_context,
        mode,
    )
    outputs.append(plan)

    if not notes:
        outputs.extend(
            _run_without_sources(
                topic,
                audience,
                purpose,
                plan,
            )
        )

    else:
        # 2. Research Synthesizer Agent
        research = _run_agent(
            "Research Synthesizer Agent",
            (
                "Extract factual claims from the supplied source notes. "
                "Return ONLY valid JSON with this exact structure:\n\n"
                "{\n"
                '  "claims": [\n'
                "    {\n"
                '      "claim": "A concise factual claim",\n'
                '      "source_reference": "Source 1 or null",\n'
                '      "supporting_excerpt": "Exact verbatim excerpt or null",\n'
                '      "evidence_status": "traceable, unverified, or unsupported",\n'
                '      "confidence": "low, medium, or high",\n'
                '      "reasoning": "Why this status was assigned"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- Use only the supplied source notes.\n"
                "- Never invent source references or excerpts.\n"
                "- Supporting excerpts must be copied exactly.\n"
                "- Do not invent missing bibliographic details.\n"
                "- If there are no factual claims, return an empty claims list.\n"
                "- Do not include Markdown fences or text outside the JSON.\n\n"
                "PLAN:\n" + plan["output"]
            ),
            context,
            mode,
        )
        outputs.append(research)

        # Validate claims deterministically in Python.
        try:
            evidence_claims = parse_evidence_output(
                research["output"],
                notes,
            )
        except ValueError as exc:
            validation_error = str(exc)
            evidence_claims = []

        validated_context = json.dumps(
            {
                "claims": evidence_claims,
                "summary": evidence_summary(evidence_claims),
                "validation_error": validation_error,
            },
            indent=2,
            ensure_ascii=False,
        )

        # 3. Analysis Agent
        analysis = _run_agent(
            "Analysis Agent",
            (
                "Analyze the validated evidence claims below. "
                "Do not add factual claims absent from this data. "
                "Treat traceable as source-text traceability only, "
                "not independent proof of truth. Clearly distinguish "
                "supported observations, inferences, and open questions. "
                "If validation failed or no claims were extracted, "
                "state that substantive analysis is limited.\n\n"
                "VALIDATED EVIDENCE DATA:\n"
                + validated_context
            ),
            metadata_context,
            mode,
        )
        outputs.append(analysis)

        # 4. Evidence Review Agent
        review = _run_agent(
            "Evidence Review Agent",
            (
                "Review every validated evidence claim and its "
                "supporting excerpt. Assess whether the excerpt "
                "actually supports the claim.\n\n"
                "Return ONLY valid JSON in this structure:\n"
                "{\n"
                '  "assessments": [\n'
                "    {\n"
                '      "claim_index": 0,\n'
                '      "support_assessment": "supported",\n'
                '      "reasoning": "Explain how the excerpt supports or fails to support the claim."\n'
                "    }\n"
                "  ],\n"
                '  "findings": [\n'
                "    {\n"
                '      "severity": "low, medium, or high",\n'
                '      "claim": "Claim or general issue",\n'
                '      "status": "Issue identified",\n'
                '      "reason": "Explain the problem",\n'
                '      "required_correction": "Suggested correction"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Assessment rules:\n"
                "- Use zero-based claim_index values.\n"
                "- Assess every supplied claim exactly once.\n"
                "- Use only these support_assessment values: "
                "supported, partially_supported, not_supported, not_assessed.\n"
                "- supported: the excerpt directly supports the complete claim.\n"
                "- partially_supported: the excerpt supports only part of the claim.\n"
                "- not_supported: the excerpt does not support the claim.\n"
                "- not_assessed: there is insufficient information to assess it.\n"
                "- Do not assess whether a claim is universally true.\n"
                "- Do not change or override Python's evidence_status.\n"
                "- Exact text matching does not automatically mean supported.\n"
                "- If there are no claims, return empty assessments and findings.\n"
                "- Do not include Markdown fences or text outside JSON.\n\n"
                "VALIDATED EVIDENCE DATA:\n"
                + validated_context
                + "\n\nANALYSIS:\n"
                + analysis["output"]
            ),
            context,
            mode,
        )
        outputs.append(review)

        # Apply semantic assessments without changing text traceability.
        try:
            evidence_claims = apply_support_assessments(
                evidence_claims,
                review["output"],
            )
        except ValueError as exc:
            support_assessment_error = str(exc)

        # Rebuild context after semantic assessment.
        validated_context = json.dumps(
            {
                "claims": evidence_claims,
                "summary": evidence_summary(evidence_claims),
                "validation_error": validation_error,
                "support_assessment_error": support_assessment_error,
            },
            indent=2,
            ensure_ascii=False,
        )

        # 5. Report Writer Agent
        final = _run_agent(
            "Report Writer Agent",
            (
                "Write a concise report using only information "
                "traceable to the supplied source material. "
                "Do not imply independent verification occurred. "
                "Clearly label inferences and limitations. "
                "Do not present unverified or unsupported claims "
                "as established findings.\n\n"
                "Include a separate evidence summary containing:\n"
                "- Exact text traceability counts.\n"
                "- Semantic support assessment counts.\n"
                "- Claims that remain not_assessed.\n"
                "- Reviewer findings and required corrections.\n\n"
                "Explain that semantic support assessment is an "
                "LLM-based assessment, not independent proof of truth. "
                "If evidence validation or semantic assessment failed, "
                "explicitly disclose that limitation.\n\n"
                "VALIDATED EVIDENCE DATA:\n"
                + validated_context
                + "\n\nANALYSIS:\n"
                + analysis["output"]
                + "\n\nREVIEW FINDINGS:\n"
                + review["output"]
            ),
            context,
            mode,
        )
        outputs.append(final)

    report = outputs[-1]["output"]
    review_output = outputs[-2]["output"]

    return {
        "topic": topic,
        "audience": audience,
        "purpose": purpose,
        "agents": outputs,
        "report": report,
        "review": review_output,
        "evidence_claims": evidence_claims,
        "evidence_summary": evidence_summary(evidence_claims),
        "evidence_validation_error": validation_error,
        "support_assessment_error": support_assessment_error,
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
        ("Planner Agent", "Demo plan: define scope, questions, evidence needs, and boundaries."),
        ("Research Synthesizer Agent", f"Demo evidence input:\n{note}"),
        ("Analysis Agent", "Demo analysis: organize themes and identify trade-offs after evidence collection."),
        ("Evidence Review Agent", "Demo review: verify all factual claims against reliable external sources."),
        ("Report Writer Agent", report),
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
        "evidence_claims": [],
        "evidence_summary": evidence_summary([]),
        "evidence_validation_error": None,
        "support_assessment_error": None,
        "total_agent_seconds": 0.0,
    }
