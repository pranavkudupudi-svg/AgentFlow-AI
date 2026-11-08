
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from agentflow.agents import run_workflow, demo_workflow
from agentflow.llm import LLMError
from agentflow.report import build_pdf

load_dotenv()

st.set_page_config(
    page_title="AgentFlow AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(115deg,#172554,#334155);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 {color:white; margin:0;}
.hero p {color:#dbeafe; margin:.4rem 0 0 0;}
.agent-card {
    padding: 1rem;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
    background: white;
}
.small-muted {color:#64748b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🧠 AgentFlow AI</h1>'
    '<p>A transparent multi-agent research and report generation system</p></div>',
    unsafe_allow_html=True,
)

st.write(
    "Enter a research topic, provide optional source notes, "
    "and watch five specialized agents collaborate."
)


# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.selectbox(
        "LLM deployment",
        [
            "Hybrid (API first, local fallback)",
            "API only",
            "Local only",
        ],
    )

    st.caption(
        "Hybrid tries your configured API first, "
        "then a local Ollama model."
    )

    if mode == "Local only":
        st.info(
            "Local CPU inference can take several minutes. "
            "Ollama timeout defaults to 600 seconds; "
            "concise source notes help."
        )

    st.divider()
    st.subheader("Backend setup")
    st.write("**API:** configure `.env` with an API key and model.")
    st.write("**Local:** install Ollama and pull a model.")
    st.write(
        "**Demo mode:** if no backend is configured, "
        "use the clearly labelled template demo."
    )

    st.divider()
    st.caption(
        "Text traceability checks whether excerpts appear in "
        "the supplied notes. Semantic support assesses whether "
        "those excerpts actually support their claims. Neither "
        "independently verifies factual truth."
    )


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

col1, col2 = st.columns([1.15, 1])

with col1:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Generative AI in higher education",
    )

    audience = st.selectbox(
        "Target audience",
        [
            "University faculty",
            "Undergraduate students",
            "General audience",
            "Business decision-makers",
        ],
    )

    purpose = st.text_area(
        "Report objective",
        placeholder=(
            "Explain the topic, evaluate benefits and risks, "
            "and present practical recommendations."
        ),
        height=90,
    )

with col2:
    source_notes = st.text_area(
        "Source notes / evidence (optional)",
        placeholder=(
            "Paste excerpts, facts, URLs with notes, "
            "or material you want the agents to analyze..."
        ),
        height=190,
        help=(
            "For stronger evidence-grounded output, paste "
            "relevant source excerpts and identify their source."
        ),
    )


# --------------------------------------------------
# WORKFLOW BUTTONS
# --------------------------------------------------

run_col, demo_col = st.columns([1, 1])

run_clicked = run_col.button(
    "🚀 Run multi-agent workflow",
    type="primary",
    use_container_width=True,
)

demo_clicked = demo_col.button(
    "🧪 Preview demo without an LLM",
    use_container_width=True,
)


if run_clicked:
    if not topic.strip():
        st.error("Please enter a research topic.")

    else:
        with st.status(
            "Agents are collaborating...",
            expanded=True,
        ) as status:

            try:
                result = run_workflow(
                    topic.strip(),
                    audience,
                    purpose.strip(),
                    source_notes.strip(),
                    mode,
                )

                status.update(
                    label="Workflow completed",
                    state="complete",
                )

                st.session_state["result"] = result

            except LLMError as exc:
                status.update(
                    label="LLM backend unavailable",
                    state="error",
                )

                st.error(str(exc))

                st.info(
                    "Check the selected backend and Ollama service. "
                    "Local CPU inference may take several minutes."
                )

            except Exception as exc:
                status.update(
                    label="Workflow failed",
                    state="error",
                )
                st.exception(exc)


if demo_clicked:
    if not topic.strip():
        st.error("Please enter a research topic.")

    else:
        st.session_state["result"] = demo_workflow(
            topic.strip(),
            audience,
            purpose.strip(),
            source_notes.strip(),
        )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.get("result")


if result:
    st.divider()

    # --------------------------------------------------
    # WORKFLOW EXECUTION
    # --------------------------------------------------

    st.subheader("Workflow execution")
    st.caption(f"Topic: {result['topic']}")

    cols = st.columns(5)

    for col, agent in zip(cols, result["agents"]):
        with col:
            st.markdown(
                f"**{agent['agent'].replace(' Agent', '')}**"
            )

            st.success("Complete")
            st.caption(agent["backend"])

            elapsed = agent.get("elapsed_seconds")

            if elapsed is not None:
                st.caption(f"Execution: {elapsed:.2f}s")

    total_time = result.get("total_agent_seconds")

    if total_time is not None and total_time > 0:
        st.caption(
            f"Total model execution time "
            f"(sum of agent calls): {total_time:.2f}s"
        )

    # --------------------------------------------------
    # EVIDENCE TRACEABILITY DASHBOARD
    # --------------------------------------------------

    st.divider()
    st.subheader("📊 Evidence Traceability Dashboard")

    claims = result.get("evidence_claims", [])
    summary = result.get("evidence_summary", {})

    total = summary.get("total", len(claims))
    traceable = summary.get("traceable", 0)
    unverified = summary.get("unverified", 0)
    unsupported = summary.get("unsupported", 0)

    coverage = (traceable / total * 100) if total else 0.0

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Total claims", total)

    with metric_cols[1]:
        st.metric("Traceable", traceable)

    with metric_cols[2]:
        st.metric("Unverified", unverified)

    with metric_cols[3]:
        st.metric("Unsupported", unsupported)

    st.markdown("#### Evidence coverage")

    st.progress(
        min(max(coverage / 100, 0.0), 1.0)
    )

    st.markdown(
        f"**{coverage:.1f}%** of extracted claims "
        "have supporting excerpts found in the supplied notes."
    )

    st.caption(
        "Coverage measures text traceability, not factual accuracy "
        "or independent verification."
    )

    validation_error = result.get("evidence_validation_error")

    if validation_error:
        st.warning(
            "Evidence validation encountered an error. "
            "The structured evidence results may be incomplete."
        )

        with st.expander("Validation error details"):
            st.code(validation_error)

    # --------------------------------------------------
    # SEMANTIC SUPPORT DASHBOARD
    # --------------------------------------------------

    st.divider()
    st.subheader("🧠 Semantic Evidence Support")

    st.caption(
        "The Evidence Reviewer assesses whether each excerpt "
        "actually supports its associated claim. This is an "
        "LLM-based assessment, not independent fact verification."
    )

    supported = summary.get("supported", 0)
    partially_supported = summary.get("partially_supported", 0)
    not_supported = summary.get("not_supported", 0)
    not_assessed = summary.get("not_assessed", 0)

    support_cols = st.columns(4)

    with support_cols[0]:
        st.metric("Supported", supported)

    with support_cols[1]:
        st.metric("Partially supported", partially_supported)

    with support_cols[2]:
        st.metric("Not supported", not_supported)

    with support_cols[3]:
        st.metric("Not assessed", not_assessed)

    support_error = result.get("support_assessment_error")

    if support_error:
        st.warning(
            "The Evidence Reviewer response could not be parsed. "
            "Claims without a valid assessment remain not assessed."
        )

        with st.expander("Semantic assessment error details"):
            st.code(support_error)

    if claims:
        assessed_count = total - not_assessed

        if total:
            assessment_coverage = assessed_count / total * 100
        else:
            assessment_coverage = 0.0

        st.markdown("#### Reviewer assessment coverage")

        st.progress(
            min(max(assessment_coverage / 100, 0.0), 1.0)
        )

        st.caption(
            f"{assessment_coverage:.1f}% of claims received "
            "a valid semantic support assessment."
        )

    # --------------------------------------------------
    # CLAIM TABLE
    # --------------------------------------------------

    if claims:
        st.divider()
        st.markdown("#### Extracted claims")

        claim_rows = []

        for index, claim in enumerate(claims, start=1):
            claim_rows.append({
                "No.": index,
                "Claim": claim.get("claim", ""),
                "Source": claim.get("source_reference") or "Not provided",
                "Text traceability": claim.get(
                    "evidence_status", "unknown"
                ).replace("_", " ").title(),
                "Semantic support": claim.get(
                    "support_assessment", "not_assessed"
                ).replace("_", " ").title(),
                "Confidence": claim.get(
                    "confidence", "unknown"
                ).title(),
            })

        st.dataframe(
            pd.DataFrame(claim_rows),
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------
        # INDIVIDUAL CLAIM DETAILS
        # --------------------------------------------------

        for index, claim in enumerate(claims, start=1):
            with st.expander(
                f"Claim {index}: {claim.get('claim', '')[:100]}"
            ):
                st.markdown("**Claim**")
                st.write(claim.get("claim", ""))

                detail_cols = st.columns(2)

                with detail_cols[0]:
                    st.write("**Text traceability**")
                    st.write(
                        claim.get(
                            "evidence_status", "unknown"
                        ).replace("_", " ").title()
                    )

                    st.write("**Source reference**")
                    st.write(
                        claim.get("source_reference") or "Not provided"
                    )

                    st.write("**Confidence**")
                    st.write(
                        claim.get("confidence", "unknown").title()
                    )

                with detail_cols[1]:
                    st.write("**Semantic support**")
                    st.write(
                        claim.get(
                            "support_assessment", "not_assessed"
                        ).replace("_", " ").title()
                    )

                    st.write("**Reviewer reasoning**")
                    st.write(
                        claim.get("support_reasoning")
                        or "No semantic assessment available."
                    )

                st.write("**Supporting excerpt**")

                excerpt = claim.get("supporting_excerpt")

                if excerpt:
                    st.info(excerpt)
                else:
                    st.write("No supporting excerpt supplied.")

                st.write("**Research Synthesizer reasoning**")
                st.write(
                    claim.get("reasoning") or "No reasoning supplied."
                )

    elif validation_error:
        st.info(
            "No structured claims are available because "
            "evidence validation failed."
        )

    else:
        st.info(
            "No structured evidence claims were extracted. "
            "This can happen when no source notes are supplied "
            "or the workflow is running in demo mode."
        )

    # --------------------------------------------------
    # AGENT OUTPUTS
    # --------------------------------------------------

    st.divider()
    st.subheader("Agent outputs")

    for i, agent in enumerate(result["agents"], start=1):
        with st.expander(
            f"{i}. {agent['agent']} · {agent['backend']}",
            expanded=(i == 5),
        ):
            st.markdown(agent["output"])

    # --------------------------------------------------
    # EVIDENCE REVIEW
    # --------------------------------------------------

    st.subheader("Review-to-report traceability")

    st.caption(
        "The Report Writer is instructed to address reviewer "
        "findings. Inspect the response section in the final report "
        "to see how each issue was handled."
    )

    if result.get("review"):
        with st.expander(
            "Evidence Reviewer findings",
            expanded=False,
        ):
            st.markdown(result["review"])

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    st.subheader("Final research report")

    st.markdown(result["report"])

    pdf = build_pdf(
        result["topic"],
        result["report"],
    )

    st.download_button(
        "⬇️ Download report as PDF",
        data=pdf,
        file_name="AgentFlow_Research_Report.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.download_button(
        "⬇️ Download report as Markdown",
        data=result["report"],
        file_name="AgentFlow_Research_Report.md",
        mime="text/markdown",
    )


else:
    st.info(
        "Your workflow results will appear here. "
        "Start with a short topic and a few source notes."
    )

    st.markdown("### The five-agent pipeline")

    st.markdown("""
    1. **Planner:** defines research questions and scope.
    2. **Research Synthesizer:** organizes supplied context into an evidence brief.
    3. **Analyst:** identifies themes, implications, and trade-offs.
    4. **Evidence Reviewer:** assesses semantic support and flags evidence issues.
    5. **Report Writer:** produces the final structured report.
    """)
