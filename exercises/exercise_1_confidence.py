"""Exercise 1 — structured PR analysis and confidence routing."""
from __future__ import annotations
import argparse
import asyncio
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from common.github import fetch_pr
from exercises.support import ExtendedReviewState, analyze, route_for

console = Console()

def node_fetch_pr(state):
    with console.status("Fetching PR from GitHub..."):
        pr = fetch_pr(state["pr_url"])
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}

async def node_analyze(state):
    with console.status("LLM reviewing the diff..."):
        analysis = await analyze(state["pr_title"], state["pr_diff"])
    console.print(f"confidence={analysis.confidence:.0%}, {len(analysis.comments)} comment(s)")
    return {"analysis": analysis, "analysis_initial": analysis, "review_version": 0, "review_kind": "initial"}

def node_route(state):
    decision = route_for(state["analysis"].confidence)
    console.print(f"decision={decision} (confidence={state['analysis'].confidence:.0%})")
    return {"decision": decision}

def terminal(name):
    def node(state):
        console.print(f"✓ {name}")
        return {"final_action": f"pending_{name}"}
    return node

def build_graph():
    graph = StateGraph(ExtendedReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", terminal("auto_approve")),
        ("human_approval", terminal("human_approval")),
        ("escalate", terminal("escalate")),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "fetch_pr")
    graph.add_edge("fetch_pr", "analyze")
    graph.add_edge("analyze", "route")
    graph.add_conditional_edges("route", lambda state: state["decision"], {
        "auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate",
    })
    for node in ("auto_approve", "human_approval", "escalate"):
        graph.add_edge(node, END)
    return graph.compile()

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()
    result = asyncio.run(build_graph().ainvoke({"pr_url": args.pr}))
    console.print(f"confidence={result['analysis'].confidence:.0%}; action={result['final_action']}")

if __name__ == "__main__":
    main()
