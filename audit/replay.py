"""Replay structured audit events and list recent review sessions."""
from __future__ import annotations

import argparse
import asyncio
from rich.console import Console
from rich.table import Table
from common.db import db_conn, replay_events

RISK_COLOR = {"low": "green", "med": "yellow", "high": "red"}

async def list_threads() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MIN(timestamp) AS started, MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                     WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC LIMIT 25
            """
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]

async def replay(thread_id: str) -> None:
    console = Console()
    events = await replay_events(thread_id)
    if not events:
        console.print(f"[red]No events found for thread {thread_id}[/red]")
        return
    console.rule(f"[bold]Replay {thread_id}")
    for event in events:
        risk = event["risk_level"]
        reason = (event["reason"] or "")[:60]
        reviewer = event["reviewer_id"] or "-"
        console.print(
            f"[dim]{event['timestamp']}[/dim]  [cyan]{event['action']:<18}[/cyan] "
            f"conf=[bold]{event['confidence']:.2f}[/bold] "
            f"risk=[{RISK_COLOR.get(risk, 'white')}]{risk:<4}[/] "
            f"decision=[magenta]{event['decision']:<10}[/magenta] "
            f"reviewer={reviewer:<14} {event['execution_time_ms']:>5}ms  [dim]{reason}[/dim]"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", help="Replay one thread by id")
    parser.add_argument("--list", action="store_true", help="List recent threads")
    args = parser.parse_args()
    if args.list:
        console = Console()
        table = Table(title="Recent review sessions")
        for column in ("thread_id", "pr_url", "started", "last_event", "worst_risk", "events"):
            table.add_column(column)
        for row in asyncio.run(list_threads()):
            table.add_row(*(str(row[key]) for key in ("thread_id", "pr_url", "started", "last_event", "worst_risk", "events")))
        console.print(table)
    elif args.thread:
        asyncio.run(replay(args.thread))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
