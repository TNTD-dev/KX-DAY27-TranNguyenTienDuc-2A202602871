"""Exercise 2 — pause for an explicit reviewer decision."""
from __future__ import annotations
import argparse
import asyncio
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel
from common.github import fetch_pr
from exercises.support import ExtendedReviewState, analyze, post_approved_review, route_for

console = Console()

async def node_fetch_pr(state):
    pr = fetch_pr(state["pr_url"])
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}

async def node_analyze(state):
    a = await analyze(state["pr_title"], state["pr_diff"])
    return {"analysis": a, "analysis_initial": a, "review_version": 0, "review_kind": "initial"}

def node_route(state):
    return {"decision": route_for(state["analysis"].confidence)}

async def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000], "review_version": state.get("review_version", 0),
        "review_kind": state.get("review_kind", "initial"), "risk_factors": a.risk_factors,
    })
    if not isinstance(response, dict) or response.get("choice") not in {"approve", "reject", "edit"}:
        raise ValueError("Reviewer response must contain approve, reject, or edit")
    if response["choice"] == "edit" and not str(response.get("feedback", "")).strip():
        raise ValueError("Feedback is required for Edit")
    return {"human_choice": response["choice"], "human_feedback": response.get("feedback", "")}

async def node_commit(state):
    if state.get("human_choice") == "approve":
        return await post_approved_review(state)
    return {"final_action": "rejected"}

async def node_auto_approve(state):
    result = await post_approved_review({**state, "human_choice": "approve"})
    action = result.get("final_action")
    result["final_action"] = "auto_committed" if action == "committed" else (
        "auto_simulation_approved" if action == "simulation_approved" else "auto_commit_failed"
    )
    return result

async def node_escalate(state):
    return {"final_action": "pending_escalation"}

def build_graph():
    graph = StateGraph(ExtendedReviewState)
    for name, fn in [("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
                     ("human_approval", node_human_approval), ("commit", node_commit),
                     ("auto_approve", node_auto_approve), ("escalate", node_escalate)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "fetch_pr"); graph.add_edge("fetch_pr", "analyze"); graph.add_edge("analyze", "route")
    graph.add_conditional_edges("route", lambda s: s["decision"], {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"})
    graph.add_edge("human_approval", "commit"); graph.add_edge("commit", END)
    graph.add_edge("auto_approve", END); graph.add_edge("escalate", END)
    return graph.compile(checkpointer=MemorySaver())

def prompt_human(payload):
    console.print(Panel.fit(f"{payload['confidence']:.0%} · {payload['review_kind']}\n{payload['confidence_reasoning']}\n\n{payload['summary']}", title="Approval request", border_style="green"))
    for c in payload["comments"]: console.print(f"[{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
    console.print(payload["diff_preview"])
    choice = console.input("Choice (approve/reject/edit): ").strip().lower()
    feedback = console.input("Edit instructions: ").strip() if choice == "edit" else ""
    return {"choice": choice, "feedback": feedback}

async def run(pr_url: str):
    app = build_graph(); thread_id = str(uuid.uuid4()); cfg = {"configurable": {"thread_id": thread_id}}
    result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = await app.ainvoke(Command(resume=prompt_human(result["__interrupt__"][0].value)), cfg)
    console.print(result.get("final_action"))

def main():
    load_dotenv(); parser = argparse.ArgumentParser(); parser.add_argument("--pr", required=True); args = parser.parse_args()
    asyncio.run(run(args.pr))
if __name__ == "__main__": main()
