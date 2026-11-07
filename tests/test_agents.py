from agentflow.agents import demo_workflow

def test_demo_workflow_has_five_agents():
    result = demo_workflow("AI in education", "Students", "Explore benefits")
    assert len(result["agents"]) == 5
    assert "AI in education" in result["report"]
    assert all(a["backend"].startswith("DEMO") for a in result["agents"])
