
import json
import pytest

from agentflow.agents import demo_workflow
from agentflow.evidence import (
    apply_support_assessments,
    evidence_summary,
    parse_evidence_output,
    validate_claim,
)


# --------------------------------------------------
# Demo workflow
# --------------------------------------------------

def test_demo_workflow_has_five_agents():
    result = demo_workflow(
        "AI in education",
        "Students",
        "Explore benefits",
    )

    assert len(result["agents"]) == 5
    assert "AI in education" in result["report"]
    assert all(
        agent["backend"].startswith("DEMO")
        for agent in result["agents"]
    )


# --------------------------------------------------
# Evidence validation
# --------------------------------------------------

def sample_claim(
    status="traceable",
    excerpt="The pilot programme allowed students to use AI.",
):
    return {
        "claim": "The pilot programme allowed students to use AI.",
        "source_reference": "Source 1",
        "supporting_excerpt": excerpt,
        "evidence_status": status,
        "confidence": "high",
        "reasoning": "The source describes the pilot programme.",
    }


def test_traceable_claim_with_exact_excerpt():
    notes = "Source 1: The pilot programme allowed students to use AI."

    result = validate_claim(sample_claim(), notes)

    assert result.evidence_status == "traceable"


def test_invented_excerpt_is_downgraded():
    notes = "Source 1: Students participated in a pilot programme."

    result = validate_claim(sample_claim(), notes)

    assert result.evidence_status == "unverified"
    assert result.confidence == "low"


def test_non_traceable_claim_is_not_promoted():
    notes = "Source 1: The pilot programme allowed students to use AI."

    claim = sample_claim(status="unverified")
    result = validate_claim(claim, notes)

    assert result.evidence_status == "unverified"


def test_invalid_claim_status_is_rejected():
    claim = sample_claim()
    claim["evidence_status"] = "verified"

    with pytest.raises(ValueError):
        validate_claim(claim, "Some source notes.")


def test_parse_evidence_output():
    notes = "Source 1: The pilot programme allowed students to use AI."

    output = json.dumps({
        "claims": [sample_claim()]
    })

    claims = parse_evidence_output(output, notes)

    assert len(claims) == 1
    assert claims[0]["evidence_status"] == "traceable"
    assert claims[0]["support_assessment"] == "not_assessed"


def test_invalid_evidence_json_is_rejected():
    with pytest.raises(ValueError):
        parse_evidence_output("This is not JSON", "Source notes.")


# --------------------------------------------------
# Evidence summary
# --------------------------------------------------

def test_evidence_summary_counts_statuses():
    claims = [
        {"evidence_status": "traceable"},
        {"evidence_status": "traceable"},
        {"evidence_status": "unverified"},
        {"evidence_status": "unsupported"},
    ]

    summary = evidence_summary(claims)

    assert summary == {
        "total": 4,
        "traceable": 2,
        "unverified": 1,
        "unsupported": 1,
        "supported": 0,
        "partially_supported": 0,
        "not_supported": 0,
        "not_assessed": 4,
    }


# --------------------------------------------------
# Semantic support assessments
# --------------------------------------------------

def test_apply_supported_assessment():
    claims = [{
        "claim": "Students used AI for brainstorming.",
        "evidence_status": "traceable",
        "support_assessment": "not_assessed",
    }]

    reviewer_output = json.dumps({
        "assessments": [{
            "claim_index": 0,
            "support_assessment": "supported",
            "reasoning": "The excerpt directly describes AI brainstorming.",
        }]
    })

    result = apply_support_assessments(claims, reviewer_output)

    assert result[0]["support_assessment"] == "supported"
    assert "directly describes" in result[0]["support_reasoning"]


def test_apply_multiple_support_assessments():
    claims = [
        {"claim": "Claim one", "support_assessment": "not_assessed"},
        {"claim": "Claim two", "support_assessment": "not_assessed"},
    ]

    reviewer_output = json.dumps({
        "assessments": [
            {
                "claim_index": 0,
                "support_assessment": "supported",
                "reasoning": "Directly supported by the excerpt.",
            },
            {
                "claim_index": 1,
                "support_assessment": "partially_supported",
                "reasoning": "The excerpt supports only part of the claim.",
            },
        ]
    })

    result = apply_support_assessments(claims, reviewer_output)

    assert result[0]["support_assessment"] == "supported"
    assert result[1]["support_assessment"] == "partially_supported"


def test_invalid_support_assessment_is_rejected():
    claims = [{"claim": "Claim one"}]

    reviewer_output = json.dumps({
        "assessments": [{
            "claim_index": 0,
            "support_assessment": "verified",
            "reasoning": "Invalid assessment label.",
        }]
    })

    with pytest.raises(ValueError):
        apply_support_assessments(claims, reviewer_output)


def test_duplicate_assessment_index_is_rejected():
    claims = [{"claim": "Claim one"}]

    reviewer_output = json.dumps({
        "assessments": [
            {
                "claim_index": 0,
                "support_assessment": "supported",
                "reasoning": "First assessment.",
            },
            {
                "claim_index": 0,
                "support_assessment": "not_supported",
                "reasoning": "Duplicate assessment.",
            },
        ]
    })

    with pytest.raises(ValueError):
        apply_support_assessments(claims, reviewer_output)
