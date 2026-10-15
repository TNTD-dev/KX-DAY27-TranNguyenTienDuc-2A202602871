from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from streamlit.testing.v1 import AppTest

from common.db import replay_events, write_audit_event
from common.schemas import AuditEntry, PRAnalysis, ReviewComment, risk_level_for
from exercises import exercise_1_confidence as e1
from exercises import exercise_2_hitl as e2
from exercises import exercise_3_escalation as e3
from exercises import exercise_4_audit as e4
from exercises.exercise_4_audit import retry_failed_commit
from exercises.dashboard import checkpoint_history, confidence_statistics, fork_approval_checkpoint, recent_sessions
from exercises.support import first_interrupt_value, post_approved_review, post_comment_once, render_comment, route_for, open_checkpointer


PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"


def analysis(confidence: float, question: bool = False) -> PRAnalysis:
    return PRAnalysis(
        summary="Adds a task field.",
        risk_factors=["Migration needs confirmation"],
        comments=[ReviewComment(file="src/tasks.py", line=12, severity="issue", body="Validate priority.")],
        confidence=confidence,
        confidence_reasoning="The diff is small; migration context is unknown.",
        escalation_questions=["Is the new priority column migrated for existing rows?"] if question else [],
    )


def fake_pr():
    return SimpleNamespace(
        title="Add priority", diff="diff --git a/src/tasks.py b/src/tasks.py\n+priority",
        files_changed=["src/tasks.py"], head_sha="abc123456", author="author",
    )


class FakeStructuredLLM:
    def __init__(self, *values):
        self.values = list(values)

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        if not self.values:
            raise AssertionError("Unexpected LLM call")
        return self.values.pop(0)


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_threshold_edges(self):
        self.assertEqual(route_for(.579), "escalate")
        self.assertEqual(route_for(.58), "human_approval")
        self.assertEqual(route_for(.729), "human_approval")
        self.assertEqual(route_for(.73), "auto_approve")

    async def test_exercise_one_routes_without_live_services(self):
        for score, expected in ((.4, "escalate"), (.65, "human_approval"), (.9, "auto_approve")):
            with self.subTest(score=score), patch.object(e1, "fetch_pr", return_value=fake_pr()), \
                 patch.object(e1, "analyze", new=AsyncMock(return_value=analysis(score))):
                result = await e1.build_graph().ainvoke({"pr_url": PR_URL})
                self.assertEqual(result["decision"], expected)


class CheckpointFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tempdir.name) / "lab.db")
        self.env_patch = patch.dict(os.environ, {"HITL_DB_PATH": self.db, "GITHUB_USER": "reviewer"})
        self.env_patch.start()

    async def asyncTearDown(self):
        self.env_patch.stop()
        self.tempdir.cleanup()

    async def _new_audit_graph(self, fake_analysis, llm_values=()):
        patcher_fetch = patch.object(e4, "fetch_pr", return_value=fake_pr())
        patcher_analyze = patch.object(e4, "analyze", new=AsyncMock(return_value=fake_analysis))
        patcher_llm = patch.object(e4, "get_llm", return_value=FakeStructuredLLM(*llm_values))
        patcher_post = patch.object(e4, "post_approved_review", new=AsyncMock(
            return_value={"final_action": "committed", "posted_comment_url": "https://github.com/example/comment/1", "approved_version": 0},
        ))
        for p in (patcher_fetch, patcher_analyze, patcher_llm, patcher_post):
            p.start()
            self.addCleanup(p.stop)
        return patcher_post.new

    async def test_durable_resume_and_completed_thread_is_not_restarted(self):
        post = await self._new_audit_graph(analysis(.65))
        thread = "resume-test"
        cfg = {"configurable": {"thread_id": thread}}
        async with open_checkpointer(self.db) as cp:
            graph = e4.build_graph(cp)
            paused = await graph.ainvoke({"pr_url": PR_URL, "thread_id": thread}, cfg)
            self.assertIn("__interrupt__", paused)
        async with open_checkpointer(self.db) as cp:
            graph = e4.build_graph(cp)
            restored = await e4.invoke_or_resume(graph, PR_URL, thread, is_resume=True)
            self.assertIn("__interrupt__", restored)
            final = await graph.ainvoke(Command(resume={"choice": "approve", "feedback": ""}), cfg)
            self.assertEqual(final["final_action"], "committed")
            again = await e4.invoke_or_resume(graph, PR_URL, thread, is_resume=True)
            self.assertEqual(again["final_action"], "committed")
        post.assert_awaited_once()
        events = await replay_events(thread)
        self.assertTrue(any(x["action"] == "human_approval" and x["decision"] == "approve" for x in events))
        self.assertTrue(all(x["reviewer_id"] == "reviewer" for x in events if x["decision"] == "approve"))

    async def test_escalation_requires_final_approval_and_can_fork_simulation(self):
        refined = analysis(.64)
        post = await self._new_audit_graph(analysis(.4, question=True), (refined,))
        parent = "escalation-test"
        cfg = {"configurable": {"thread_id": parent}}
        async with open_checkpointer(self.db) as cp:
            graph = e4.build_graph(cp)
            paused = await graph.ainvoke({"pr_url": PR_URL, "thread_id": parent}, cfg)
            self.assertEqual(paused["__interrupt__"][0].value["kind"], "escalation")
            refined_wait = await graph.ainvoke(Command(resume={
                "Is the new priority column migrated for existing rows?": "Yes, migration backfills zero.",
            }), cfg)
            self.assertEqual(refined_wait["__interrupt__"][0].value["review_kind"], "refined")
            history = await checkpoint_history(cp, parent)
            eligible = next(
                item for item in history
                if item["has_pending_interrupt"] and "human_approval" in item["next"]
            )
            child, fork_result = await fork_approval_checkpoint(cp, parent, eligible["config"])
            self.assertEqual(fork_result["__interrupt__"][0].value["kind"], "approval_request")
            final = await graph.ainvoke(Command(resume={"choice": "reject", "feedback": ""}), {"configurable": {"thread_id": child}})
            self.assertEqual(final["final_action"], "rejected")
            self.assertTrue(final["simulation"])
            self.assertEqual(final["parent_thread_id"], parent)
        post.assert_not_awaited()
        events = await replay_events(parent)
        self.assertTrue(any(x["action"] == "synthesize" and x["confidence"] == refined.confidence for x in events))

    async def test_edit_creates_new_version_and_posts_only_after_reapproval(self):
        edited = analysis(.68)
        post = await self._new_audit_graph(analysis(.65), (edited,))
        thread = "edit-test"
        cfg = {"configurable": {"thread_id": thread}}
        async with open_checkpointer(self.db) as cp:
            graph = e4.build_graph(cp)
            first = await graph.ainvoke({"pr_url": PR_URL, "thread_id": thread}, cfg)
            second = await graph.ainvoke(Command(resume={"choice": "edit", "feedback": "Mention migration compatibility."}), cfg)
            self.assertEqual(second["__interrupt__"][0].value["review_kind"], "edited")
            self.assertEqual(second["__interrupt__"][0].value["review_version"], 1)
            final = await graph.ainvoke(Command(resume={"choice": "approve", "feedback": ""}), cfg)
        self.assertEqual(final["final_action"], "committed")
        post.assert_awaited_once()
        self.assertIn("hitl-pr-review:edit-test:1", render_comment(final))

    async def test_failed_approved_post_can_be_safely_retried(self):
        for p in (
            patch.object(e4, "fetch_pr", return_value=fake_pr()),
            patch.object(e4, "analyze", new=AsyncMock(return_value=analysis(.65))),
        ):
            p.start()
            self.addCleanup(p.stop)
        thread = "retry-test"
        cfg = {"configurable": {"thread_id": thread}}
        with patch("exercises.support.post_comment_once", side_effect=[
            RuntimeError("temporary network failure"), "https://github.com/comment/2",
        ]) as post:
            async with open_checkpointer(self.db) as cp:
                graph = e4.build_graph(cp)
                await graph.ainvoke({"pr_url": PR_URL, "thread_id": thread}, cfg)
                failed = await graph.ainvoke(
                    Command(resume={"choice": "approve", "feedback": ""}), cfg,
                )
                self.assertEqual(failed["final_action"], "commit_failed")
                self.assertIn("temporary network failure", failed["commit_error"])
                final = await retry_failed_commit(cp, thread)
        self.assertEqual(final["final_action"], "committed")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].args[2], post.call_args_list[1].args[2])

    async def test_approval_rejection_never_posts(self):
        await self._new_audit_graph(analysis(.6))
        with patch.object(e4, "post_approved_review", new=AsyncMock()) as post:
            thread = "reject-test"
            cfg = {"configurable": {"thread_id": thread}}
            async with open_checkpointer(self.db) as cp:
                graph = e4.build_graph(cp)
                await graph.ainvoke({"pr_url": PR_URL, "thread_id": thread}, cfg)
                final = await graph.ainvoke(Command(resume={"choice": "reject", "feedback": ""}), cfg)
            self.assertEqual(final["final_action"], "rejected")
            post.assert_not_awaited()


class SafetyAndDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_restored_interrupt_payload_and_missing_post_url(self):
        payload = {"kind": "approval_request"}
        self.assertEqual(first_interrupt_value({"__interrupt__": [{"value": payload}]}), payload)
        self.assertEqual(first_interrupt_value({"__interrupt__": [SimpleNamespace(value=payload)]}), payload)
        with patch("exercises.support.find_existing_comment", return_value=None), \
             patch("exercises.support._headers", return_value={}), \
             patch("exercises.support.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value.json.return_value = {}
            with self.assertRaisesRegex(RuntimeError, "did not return a URL"):
                post_comment_once(PR_URL, "body", "marker")

    async def test_idempotency_lookup_skips_second_post(self):
        with patch("exercises.support.find_existing_comment", return_value="https://github.com/comment/1"), \
             patch("exercises.support.httpx.Client") as client:
            url = post_comment_once(PR_URL, "body", "marker")
        self.assertEqual(url, "https://github.com/comment/1")
        client.assert_not_called()

    async def test_session_risk_order_and_confidence_statistics(self):
        tempdir = tempfile.TemporaryDirectory()
        database = str(Path(tempdir.name) / "summary.db")
        def entry(action, confidence, decision, reason=None):
            return AuditEntry(
                agent_id="test", action=action, confidence=confidence,
                risk_level=risk_level_for(confidence), decision=decision,
                reason=reason, execution_time_ms=5,
            )
        with patch.dict(os.environ, {"HITL_DB_PATH": database}):
            await write_audit_event(thread_id="med-thread", pr_url=PR_URL, entry=entry("analyze", .65, "pending"))
            await write_audit_event(thread_id="med-thread", pr_url=PR_URL, entry=entry("route", .65, "pending", "route=human_approval"))
            await write_audit_event(thread_id="med-thread", pr_url=PR_URL, entry=entry("analyze", .4, "pending"))
            await write_audit_event(thread_id="high-thread", pr_url=PR_URL, entry=entry("analyze", .4, "pending"))
            await write_audit_event(thread_id="high-thread", pr_url=PR_URL, entry=entry("route", .4, "escalate", "route=escalate"))
            sessions = await recent_sessions()
            stats = await confidence_statistics()
        by_thread = {row["thread_id"]: row for row in sessions}
        self.assertEqual(by_thread["high-thread"]["worst_risk"], "high")
        self.assertEqual(stats["sessions"], 2)
        self.assertEqual(stats["branches"]["human_approval"], 1)
        self.assertEqual(stats["branches"]["escalate"], 1)
        self.assertEqual(stats["buckets"]["58–72%"]["sessions"], 1)
        tempdir.cleanup()

    async def test_failed_post_is_not_reported_as_success(self):
        state = {"analysis": analysis(.7), "thread_id": "t", "review_version": 0, "pr_url": PR_URL}
        with patch("exercises.support.post_comment_once", side_effect=RuntimeError("GitHub unavailable")):
            result = await post_approved_review(state)
        self.assertEqual(result["final_action"], "commit_failed")
        self.assertIn("unavailable", result["commit_error"])

    async def test_app_smoke_and_stats_query(self):
        tempdir = tempfile.TemporaryDirectory()
        with patch.dict(os.environ, {"HITL_DB_PATH": str(Path(tempdir.name) / "ui.db")}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            stats = await confidence_statistics()
        self.assertFalse(app.exception)
        self.assertTrue(any("HITL PR Review Agent" in title.value for title in app.title))
        self.assertEqual(stats["sessions"], 0)
        tempdir.cleanup()


if __name__ == "__main__":
    unittest.main()


class StreamlitInteractionTests(unittest.TestCase):
    def test_sample_pr_button_starts_review_with_its_url(self):
        tempdir = tempfile.TemporaryDirectory()
        try:
            db = str(Path(tempdir.name) / "sample-ui.db")
            requested_urls = []

            class FakeGraph:
                async def ainvoke(self, value, _cfg):
                    requested_urls.append(value["pr_url"])
                    return {"final_action": "rejected", "pr_url": value["pr_url"]}

            with patch.dict(os.environ, {"HITL_DB_PATH": db}), \
                 patch("exercises.exercise_4_audit.build_graph", return_value=FakeGraph()):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
                app.button(key="sample-pr-2").click().run()
                self.assertEqual(requested_urls, ["https://github.com/VinUni-AI20k/PR-Demo/pull/2"])
                self.assertTrue(any("Review bị từ chối" in item.value for item in app.warning))
                self.assertFalse(app.exception)
        finally:
            tempdir.cleanup()

    def test_approval_buttons_resume_with_expected_payload(self):
        tempdir = tempfile.TemporaryDirectory()
        try:
            db = str(Path(tempdir.name) / "approval-ui.db")
            approval = {
                "kind": "approval_request", "confidence": .65,
                "confidence_reasoning": "One migration detail needs a reviewer.",
                "summary": "Adds priority.", "comments": [], "diff_preview": "+priority",
                "review_version": 0, "review_kind": "initial", "risk_factors": [],
            }

            class FakeGraph:
                async def ainvoke(self, value, _cfg):
                    if isinstance(value, Command):
                        return {
                            "final_action": "committed", "pr_url": PR_URL,
                            "analysis": analysis(.65), "posted_comment_url": "https://github.com/comment/1",
                        }
                    return {"__interrupt__": [SimpleNamespace(value=approval)]}

            with patch.dict(os.environ, {"HITL_DB_PATH": db}), \
                 patch("exercises.exercise_4_audit.build_graph", return_value=FakeGraph()):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
                app.text_input[0].set_value(PR_URL)
                app.button(key="FormSubmitter:start_review-Run review").click().run()
                self.assertTrue(any("Approval requested" in item.value for item in app.subheader))
                app.text_area[0].set_value("")
                app.button(key="approve-0").click().run()
                self.assertTrue(any("đã được đăng" in item.value for item in app.success))
                self.assertFalse(app.exception)
        finally:
            tempdir.cleanup()

    def test_escalation_form_requires_answers_and_resumes(self):
        tempdir = tempfile.TemporaryDirectory()
        try:
            db = str(Path(tempdir.name) / "escalation-ui.db")
            escalation = {
                "kind": "escalation", "confidence": .4,
                "confidence_reasoning": "The migration intent is unclear.",
                "summary": "Changes task priority.", "risk_factors": ["Migration"],
                "questions": ["How are existing rows migrated?"],
            }
            refined = {
                "kind": "approval_request", "confidence": .64,
                "confidence_reasoning": "The answer clarified one assumption.",
                "summary": "Refined review.", "comments": [], "diff_preview": "+priority",
                "review_version": 1, "review_kind": "refined", "risk_factors": [],
            }

            class FakeGraph:
                async def ainvoke(self, value, _cfg):
                    if isinstance(value, Command):
                        return {"__interrupt__": [SimpleNamespace(value=refined)]}
                    return {"__interrupt__": [SimpleNamespace(value=escalation)]}

            with patch.dict(os.environ, {"HITL_DB_PATH": db}), \
                 patch("exercises.exercise_4_audit.build_graph", return_value=FakeGraph()):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
                app.text_input[0].set_value(PR_URL)
                app.button(key="FormSubmitter:start_review-Run review").click().run()
                self.assertTrue(any("Strong escalation" in item.value for item in app.subheader))
                app.text_input(key="answer-" + hashlib.sha256(escalation["questions"][0].encode()).hexdigest()[:12]).set_value("Backfill uses zero.")
                app.button(key="FormSubmitter:escalation_answers-Gửi câu trả lời").click().run()
                self.assertTrue(any("reviewed" in item.value or "refined" in item.value.lower() for item in app.subheader))
                self.assertFalse(app.exception)
        finally:
            tempdir.cleanup()
