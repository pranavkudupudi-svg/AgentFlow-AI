
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional


VALID_STATUSES = {
    "traceable",
    "unverified",
    "unsupported",
}

VALID_CONFIDENCE = {
    "low",
    "medium",
    "high",
}

VALID_SUPPORT_ASSESSMENTS = {
    "supported",
    "partially_supported",
    "not_supported",
    "not_assessed",
}


@dataclass
class EvidenceClaim:
    claim: str
    source_reference: Optional[str]
    supporting_excerpt: Optional[str]
    evidence_status: str
    confidence: str
    reasoning: str
    support_assessment: str = "not_assessed"
    support_reasoning: str = (
        "Semantic support assessment has not yet been performed."
    )

    def to_dict(self):
        return asdict(self)


def validate_claim(claim, source_notes):
    if not isinstance(claim, dict):
        raise ValueError("Each claim must be a JSON object.")

    required_fields = {
        "claim",
        "source_reference",
        "supporting_excerpt",
        "evidence_status",
        "confidence",
        "reasoning",
    }

    missing = required_fields - claim.keys()
    if missing:
        raise ValueError(
            f"Missing claim fields: {', '.join(sorted(missing))}"
        )

    text = claim["claim"]
    reference = claim["source_reference"]
    excerpt = claim["supporting_excerpt"]
    status = claim["evidence_status"]
    confidence = claim["confidence"]
    reasoning = claim["reasoning"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Claim text cannot be empty.")

    if reference is not None and not isinstance(reference, str):
        raise ValueError("Source reference must be a string or null.")

    if excerpt is not None and not isinstance(excerpt, str):
        raise ValueError("Supporting excerpt must be a string or null.")

    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid evidence status: {status}")

    if confidence not in VALID_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {confidence}")

    if not isinstance(reasoning, str):
        raise ValueError("Reasoning must be a string.")

    # Normalize whitespace only for matching.
    normalized_excerpt = re.sub(r"\s+", " ", excerpt or "").strip()
    normalized_notes = re.sub(r"\s+", " ", source_notes or "").strip()

    if (
        status == "traceable"
        and (
            not reference
            or not normalized_excerpt
            or normalized_excerpt not in normalized_notes
        )
    ):
        status = "unverified"
        confidence = "low"
        reasoning = (
            "The supporting excerpt could not be verified "
            "in the original source notes."
        )

    if status == "unsupported":
        confidence = "low"

    return EvidenceClaim(
        claim=text.strip(),
        source_reference=reference,
        supporting_excerpt=excerpt,
        evidence_status=status,
        confidence=confidence,
        reasoning=reasoning.strip(),
    )


def _extract_json_object(output):
    if not isinstance(output, str):
        raise ValueError("Model output must be text.")

    text = output.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No complete JSON object found.")

    return text[start:end + 1]


def _parse_model_json(output):
    json_text = _extract_json_object(output)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned invalid JSON: {exc}"
        ) from exc


def parse_evidence_output(output, source_notes=""):
    data = _parse_model_json(output)

    if not isinstance(data, dict):
        raise ValueError("Evidence output must be a JSON object.")

    claims = data.get("claims")

    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list.")

    return [
        validate_claim(claim, source_notes).to_dict()
        for claim in claims
    ]


def apply_support_assessments(claims, reviewer_output):
    data = _parse_model_json(reviewer_output)

    if not isinstance(data, dict):
        raise ValueError("Reviewer output must be a JSON object.")

    assessments = data.get("assessments")

    if not isinstance(assessments, list):
        raise ValueError("'assessments' must be a list.")

    updated_claims = [dict(claim) for claim in claims]
    seen_indices = set()

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("Each assessment must be a JSON object.")

        index = assessment.get("claim_index")
        support = assessment.get("support_assessment")
        reasoning = assessment.get("reasoning")

        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("claim_index must be an integer.")

        if index < 0 or index >= len(updated_claims):
            raise ValueError(f"Invalid claim_index: {index}")

        if index in seen_indices:
            raise ValueError(f"Duplicate claim_index: {index}")

        if support not in VALID_SUPPORT_ASSESSMENTS:
            raise ValueError(f"Invalid support assessment: {support}")

        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Assessment reasoning cannot be empty.")

        seen_indices.add(index)
        updated_claims[index]["support_assessment"] = support
        updated_claims[index]["support_reasoning"] = reasoning.strip()

    return updated_claims


def evidence_summary(claims):
    summary = {
        "total": len(claims),
        "traceable": 0,
        "unverified": 0,
        "unsupported": 0,
        "supported": 0,
        "partially_supported": 0,
        "not_supported": 0,
        "not_assessed": 0,
    }

    for claim in claims:
        traceability = claim.get("evidence_status", "unverified")
        support = claim.get("support_assessment", "not_assessed")

        if traceability in summary:
            summary[traceability] += 1

        if support in summary:
            summary[support] += 1

    return summary
