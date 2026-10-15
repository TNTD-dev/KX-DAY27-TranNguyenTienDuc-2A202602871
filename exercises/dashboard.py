"""Read-only audit summaries and safe checkpoint simulation helpers."""
from __future__ import annotations

import uuid
from collections import Counter, defaultdict

from common.db import db_conn
from exercises.exercise_4_audit import build_graph


async def recent_sessions(limit: int = 20) -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                     WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]


async def confidence_statistics() -> dict:
    async with db_conn() as conn:
        async with conn.execute(
            """
            WITH initial AS (
                SELECT thread_id, MIN(id) AS event_id
                  FROM audit_events WHERE action = 'analyze'
                 GROUP BY thread_id
            ),
            routed AS (
                SELECT thread_id,
                       CASE
                         WHEN reason LIKE '%route=auto_approve%' THEN 'auto_approve'
                         WHEN reason LIKE '%route=human_approval%' THEN 'human_approval'
                         WHEN reason LIKE '%route=escalate%' THEN 'escalate'
                         ELSE 'incomplete'
                       END AS decision,
                       ROW_NUMBER() OVER (PARTITION BY thread_id ORDER BY id DESC) AS n
                  FROM audit_events WHERE action = 'route'
            ),
            human AS (
                SELECT thread_id, decision,
                       ROW_NUMBER() OVER (PARTITION BY thread_id ORDER BY id DESC) AS n
                  FROM audit_events
                 WHERE action = 'human_approval' AND decision IN ('approve','reject','edit')
            ),
            duration AS (
                SELECT thread_id, SUM(execution_time_ms) AS graph_ms
                  FROM audit_events GROUP BY thread_id
            )
            SELECT a.thread_id, a.confidence, r.decision AS branch,
                   h.decision AS human_choice, d.graph_ms
              FROM initial i
              JOIN audit_events a ON a.id = i.event_id
              LEFT JOIN routed r ON r.thread_id = i.thread_id AND r.n = 1
              LEFT JOIN human h ON h.thread_id = i.thread_id AND h.n = 1
              LEFT JOIN duration d ON d.thread_id = i.thread_id
            """
        ) as cur:
            rows = [dict(row) for row in await cur.fetchall()]

    branches = Counter(row["branch"] or "incomplete" for row in rows)
    buckets: dict[str, dict] = defaultdict(lambda: {"approve": 0, "reject": 0, "edit": 0, "sessions": 0})
    for row in rows:
        label = "<58%" if row["confidence"] < .58 else ("58–72%" if row["confidence"] < .73 else "≥73%")
        buckets[label]["sessions"] += 1
        if row["human_choice"]:
            buckets[label][row["human_choice"]] += 1
    return {
        "sessions": len(rows),
        "branches": dict(branches),
        "buckets": dict(buckets),
        "average_graph_ms": round(sum(row["graph_ms"] or 0 for row in rows) / len(rows)) if rows else 0,
    }


async def checkpoint_history(checkpointer, thread_id: str, limit: int = 30) -> list[dict]:
    app = build_graph(checkpointer)
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshots = []
    async for snapshot in app.aget_state_history(cfg, limit=limit):
        values = snapshot.values or {}
        snapshots.append({
            "config": snapshot.config,
            "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
            "next": tuple(snapshot.next or ()),
            "created_at": snapshot.created_at,
            "decision": values.get("decision"),
            "confidence": getattr(values.get("analysis"), "confidence", None),
            "review_version": values.get("review_version"),
            "has_pending_interrupt": any(getattr(task, "interrupts", ()) for task in snapshot.tasks),
        })
    return snapshots


async def fork_approval_checkpoint(checkpointer, parent_thread_id: str, checkpoint_config: dict) -> tuple[str, dict]:
    """Fork a waiting approval into a GitHub-free simulation thread."""
    app = build_graph(checkpointer)
    parent_cfg = checkpoint_config
    snapshot = await app.aget_state(parent_cfg)
    if not snapshot.values or not any(getattr(task, "interrupts", ()) for task in snapshot.tasks):
        raise ValueError("Choose a checkpoint waiting for reviewer approval")
    if "human_approval" not in (snapshot.next or ()):
        raise ValueError("This checkpoint is not waiting for an approval decision")
    new_thread_id = str(uuid.uuid4())
    new_cfg = {"configurable": {"thread_id": new_thread_id}}
    state = dict(snapshot.values)
    state.update({
        "thread_id": new_thread_id,
        "decision": "human_approval",
        "simulation": True,
        "parent_thread_id": parent_thread_id,
        "human_choice": None,
    })
    await app.aupdate_state(new_cfg, state, as_node="route")
    result = await app.ainvoke(None, new_cfg)
    return new_thread_id, result
