"""Exercise 3 — escalation Q&A, refined approval, and reviewer edits."""
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
from exercises.support import ExtendedReviewState, analyze, get_llm, post_approved_review, route_for
from common.schemas import PRAnalysis
import time

console = Console()

async def node_fetch_pr(state):
    pr = fetch_pr(state["pr_url"])
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}

async def node_analyze(state):
    extra = (
        "When confidence is below 0.58, include 2–4 concrete escalation_questions. "
        "Each question must refer to a specific file, code path, or decision in the diff. "
        "Ask for context needed to judge the risk, for example password-hashing intent, "
        "token storage, SQL query construction, or deployment assumptions. "
        "Do not lower confidence just to force a route."
    )
    a = await analyze(state["pr_title"], state["pr_diff"], extra)
    return {"analysis": a, "analysis_initial": a, "review_version": 0, "review_kind": "initial"}

def node_route(state):
    return {"decision": route_for(state["analysis"].confidence)}

async def node_escalate(state):
    a = state["analysis"]
    questions = a.escalation_questions[:4] or [
        "What is the intended behavior for the security-sensitive code paths in this PR?",
        "Which production assumptions or migration constraints should the reviewer consider?",
    ]
    answers = interrupt({
        "kind": "escalation", "pr_url": state["pr_url"], "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning, "summary": a.summary,
        "risk_factors": a.risk_factors, "questions": questions,
    })
    if not isinstance(answers, dict) or any(not str(answers.get(q, "")).strip() for q in questions):
        raise ValueError("Answer every escalation question before resuming")
    return {"escalation_answers": answers}

async def node_synthesize(state):
    t0 = time.monotonic()
    original = state.get("analysis_initial", state["analysis"])
    qa = "\n".join(f"Q: {q}\nA: {answer}" for q, answer in state["escalation_answers"].items())
    llm = get_llm().with_structured_output(PRAnalysis)
    refined = await llm.ainvoke([
        {"role": "system", "content": "Refine the review using reviewer context. Keep unresolved risks explicit and reassess confidence honestly."},
        {"role": "user", "content": (
            f"PR title: {state['pr_title']}\nDiff:\n{state['pr_diff']}\n\n"
            f"Initial analysis:\n{original.model_dump_json()}\n\nReviewer answers:\n{qa}"
        )},
    ])
    return {
        "analysis": refined, "review_version": state.get("review_version", 0) + 1,
        "review_kind": "refined", "human_choice": None, "human_feedback": None,
    }

async def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000], "risk_factors": a.risk_factors,
        "review_version": state.get("review_version", 0),
        "review_kind": state.get("review_kind", "initial"),
    })
    if not isinstance(response, dict) or response.get("choice") not in {"approve", "reject", "edit"}:
        raise ValueError("Reviewer response must contain approve, reject, or edit")
    if response["choice"] == "edit" and not str(response.get("feedback", "")).strip():
        raise ValueError("Feedback is required for Edit")
    return {"human_choice": response["choice"], "human_feedback": str(response.get("feedback", "")).strip()}

async def node_edit_review(state):
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    current = state["analysis"]
    edited = await llm.ainvoke([
        {"role": "system", "content": "Revise your structured code review according to reviewer feedback. Preserve valid findings, keep unresolved risks explicit, and reassess confidence honestly."},
        {"role": "user", "content": (
            f"PR title: {state['pr_title']}\nDiff:\n{state['pr_diff']}\n\n"
            f"Current review:\n{current.model_dump_json()}\n\n"
            f"Reviewer edit instructions:\n{state['human_feedback']}"
        )},
    ])
    return {
        "analysis": edited, "review_version": state.get("review_version", 0) + 1,
        "review_kind": "edited", "human_choice": None, "human_feedback": None,
    }

def route_after_approval(state):
    return state["human_choice"]

async def node_commit(state):
    if state.get("human_choice") != "approve":
        return {"final_action": "rejected"}
    return await post_approved_review(state)

async def node_auto_approve(state):
    result = await post_approved_review({**state, "human_choice": "approve"})
    action = result.get("final_action")
    result["final_action"] = "auto_committed" if action == "committed" else (
        "auto_simulation_approved" if action == "simulation_approved" else "auto_commit_failed"
    )
    return result

def build_graph():
    graph = StateGraph(ExtendedReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("escalate", node_escalate),
        ("synthesize", node_synthesize), ("human_approval", node_human_approval),
        ("edit_review", node_edit_review), ("commit", node_commit),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "fetch_pr"); graph.add_edge("fetch_pr", "analyze"); graph.add_edge("analyze", "route")
    graph.add_conditional_edges("route", lambda s: s["decision"], {
        "auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate",
    })
    graph.add_edge("auto_approve", END); graph.add_edge("escalate", "synthesize")
    graph.add_edge("synthesize", "human_approval")
    graph.add_conditional_edges("human_approval", route_after_approval, {
        "approve": "commit", "reject": "commit", "edit": "edit_review",
    })
    graph.add_edge("edit_review", "human_approval"); graph.add_edge("commit", END)
    return graph.compile(checkpointer=MemorySaver())

def handle_interrupt(payload):
    if payload["kind"] == "escalation":
        console.print(Panel.fit(payload["summary"] + "\n\n" + "\n".join(payload["risk_factors"]), title="Strong escalation", border_style="yellow"))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    console.print(Panel.fit(f"{payload['confidence']:.0%} · {payload['review_kind']}\n{payload['confidence_reasoning']}\n\n{payload['summary']}", title="Approval request", border_style="green"))
    for c in payload["comments"]: console.print(f"[{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
    choice = console.input("Choice (approve/reject/edit): ").strip().lower()
    feedback = console.input("Edit instructions: ").strip() if choice == "edit" else ""
    return {"choice": choice, "feedback": feedback}

async def run(pr_url: str):
    app = build_graph(); thread_id = str(uuid.uuid4()); cfg = {"configurable": {"thread_id": thread_id}}
    result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = await app.ainvoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)
    console.print(f"final_action={result.get('final_action')}; review_version={result.get('review_version')}")

def main():
    load_dotenv(); parser = argparse.ArgumentParser(); parser.add_argument("--pr", required=True); args = parser.parse_args()
    asyncio.run(run(args.pr))
if __name__ == "__main__": main()
