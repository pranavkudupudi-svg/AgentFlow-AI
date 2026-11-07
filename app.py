import streamlit as st
from dotenv import load_dotenv
from agentflow.agents import run_workflow, demo_workflow
from agentflow.llm import LLMError
from agentflow.report import build_pdf

load_dotenv()
st.set_page_config(page_title="AgentFlow AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px;
       background: linear-gradient(115deg,#172554,#334155); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0;}
.hero p {color:#dbeafe; margin:.4rem 0 0 0;}
.agent-card {padding: 1rem; border: 1px solid #dbe3ef; border-radius: 12px; background: white;}
.small-muted {color:#64748b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🧠 AgentFlow AI</h1><p>A transparent multi-agent research and report generation system</p></div>', unsafe_allow_html=True)
st.write("Enter a research topic, provide optional source notes, and watch five specialized agents collaborate.")

with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.selectbox("LLM deployment", [
        "Hybrid (API first, local fallback)", "API only", "Local only"
    ])
    st.caption("Hybrid tries your configured API first, then a local Ollama model.")
    if mode == "Local only":
        st.info("Local CPU inference can take several minutes. Ollama timeout defaults to 600 seconds; concise source notes help.")
    st.divider()
    st.subheader("Backend setup")
    st.write("**API:** configure `.env` with an API key and model.")
    st.write("**Local:** install Ollama and pull a model.")
    st.write("**Demo mode:** if no backend is configured, use the clearly labelled template demo.")
    st.divider()
    st.caption("Evidence review checks the supplied material for issues. It does not independently verify facts.")

col1, col2 = st.columns([1.15, 1])
with col1:
    topic = st.text_input("Research topic", placeholder="e.g. Generative AI in higher education")
    audience = st.selectbox("Target audience", ["University faculty", "Undergraduate students", "General audience", "Business decision-makers"])
    purpose = st.text_area("Report objective", value="Explain the topic, evaluate benefits and risks, and present practical recommendations.", height=90)
with col2:
    source_notes = st.text_area(
        "Source notes / evidence (optional)",
        placeholder="Paste excerpts, facts, URLs with notes, or material you want the agents to analyze...",
        height=190,
        help="For stronger evidence-grounded output, paste relevant source excerpts and identify their source."
    )

run_col, demo_col = st.columns([1, 1])
run_clicked = run_col.button("🚀 Run multi-agent workflow", type="primary", use_container_width=True)
demo_clicked = demo_col.button("🧪 Preview demo without an LLM", use_container_width=True)

if run_clicked:
    if not topic.strip():
        st.error("Please enter a research topic.")
    else:
        with st.status("Agents are collaborating...", expanded=True) as status:
            try:
                result = run_workflow(topic.strip(), audience, purpose.strip(), source_notes.strip(), mode)
                status.update(label="Workflow completed", state="complete")
                st.session_state["result"] = result
            except LLMError as exc:
                status.update(label="LLM backend unavailable", state="error")
                st.error(str(exc))
                st.info("Check the selected backend and Ollama service. Local CPU inference may take several minutes.")
            except Exception as exc:
                status.update(label="Workflow failed", state="error")
                st.exception(exc)

if demo_clicked:
    if not topic.strip():
        st.error("Please enter a research topic.")
    else:
        st.session_state["result"] = demo_workflow(topic.strip(), audience, purpose.strip(), source_notes.strip())

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Workflow execution")
    st.caption(f"Topic: {result['topic']}")
    cols = st.columns(5)
    for col, agent in zip(cols, result["agents"]):
        with col:
            st.markdown(f"**{agent['agent'].replace(' Agent','')}**")
            st.success("Complete")
            st.caption(agent["backend"])
            elapsed = agent.get("elapsed_seconds")
            if elapsed is not None:
                st.caption(f"Execution: {elapsed:.2f}s")
    total_time = result.get("total_agent_seconds")
    if total_time is not None and total_time > 0:
        st.caption(f"Total model execution time (sum of agent calls): {total_time:.2f}s")

    st.subheader("Agent outputs")
    for i, agent in enumerate(result["agents"], start=1):
        with st.expander(f"{i}. {agent['agent']} · {agent['backend']}", expanded=(i == 5)):
            st.markdown(agent["output"])
    st.subheader("Review-to-report traceability")
    st.caption("The Report Writer is instructed to address every reviewer finding. Inspect the response section in the final report to see how each issue was handled.")
    if result.get("review"):
        with st.expander("Evidence Reviewer findings", expanded=False):
            st.markdown(result["review"])

    st.subheader("Final research report")
    st.markdown(result["report"])
    pdf = build_pdf(result["topic"], result["report"])
    st.download_button("⬇️ Download report as PDF", data=pdf,
                       file_name="AgentFlow_Research_Report.pdf",
                       mime="application/pdf", type="primary")
    st.download_button("⬇️ Download report as Markdown", data=result["report"],
                       file_name="AgentFlow_Research_Report.md", mime="text/markdown")
else:
    st.info("Your workflow results will appear here. Start with a short topic and a few source notes.")
    st.markdown("### The five-agent pipeline")
    st.markdown("""
    1. **Planner:** defines research questions and scope.
    2. **Research Synthesizer:** organizes supplied context into an evidence brief.
    3. **Analyst:** identifies themes, implications, and trade-offs.
    4. **Evidence Reviewer:** flags unsupported claims and missing evidence.
    5. **Report Writer:** produces the final structured report.
    """)