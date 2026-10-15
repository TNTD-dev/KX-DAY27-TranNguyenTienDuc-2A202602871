"""Streamlit approval UI for the durable HITL PR review graph.

Run with: uv run streamlit run app.py
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from common.db import db_path
from common.github import parse_pr_url
from exercises.dashboard import (
    checkpoint_history, confidence_statistics, fork_approval_checkpoint, recent_sessions,
)
from exercises.exercise_4_audit import build_graph, invoke_or_resume, retry_failed_commit
from exercises.support import first_interrupt_value, open_checkpointer

load_dotenv()
SAMPLE_PRS = {
    1: "https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    2: "https://github.com/VinUni-AI20k/PR-Demo/pull/2",
}
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")
st.caption("Phân tích PR, duyệt review và xem lại dấu vết quyết định.")

for key, default in {
    "thread_id": None, "pr_url": "", "interrupt_payload": None,
    "final": None, "loaded_thread": None, "ui_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


async def run_graph(pr_url: str, thread_id: str, resume_value=None, *, restore=False, simulation=False):
    async with open_checkpointer(db_path()) as cp:
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if restore:
            return await invoke_or_resume(graph, pr_url, thread_id, is_resume=True)
        if resume_value is None:
            return await graph.ainvoke({
                "pr_url": pr_url, "thread_id": thread_id, "simulation": simulation,
            }, cfg)
        return await graph.ainvoke(Command(resume=resume_value), cfg)


async def retry_graph(thread_id: str):
    async with open_checkpointer(db_path()) as cp:
        return await retry_failed_commit(cp, thread_id)


def receive_result(result):
    st.session_state.ui_error = None
    st.session_state.loaded_thread = st.session_state.thread_id
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = first_interrupt_value(result)
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result
        st.session_state.loaded_thread = st.session_state.thread_id
    st.query_params["thread_id"] = st.session_state.thread_id


def restore_thread(thread_id: str, pr_url: str):
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = pr_url
    st.session_state.final = None
    st.session_state.interrupt_payload = None
    try:
        with st.spinner("Khôi phục checkpoint..."):
            receive_result(asyncio.run(run_graph(pr_url, thread_id, restore=True)))
        st.session_state.loaded_thread = thread_id
    except Exception as exc:
        st.session_state.ui_error = f"Không thể khôi phục phiên: {exc}"


def render_approval_card(payload: dict) -> dict | None:
    version = payload.get("review_version", 0)
    st.subheader(f"Approval requested · {payload['confidence']:.0%} · {payload.get('review_kind', 'review')}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("simulation"):
        st.info("Chế độ thử: Approve sẽ không đăng comment GitHub.")
    st.markdown(payload["summary"])
    if payload.get("risk_factors"):
        st.warning("Rủi ro: " + "; ".join(payload["risk_factors"]))
    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** {comment['file']}:{comment.get('line') or '?'} — "
            f"{comment['body']}"
        )
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_area(
        "Ghi chú hoặc hướng dẫn sửa (bắt buộc khi chọn Edit)",
        key=f"feedback-{version}",
    )
    approve, reject, edit = st.columns(3)
    if approve.button("Approve", type="primary", key=f"approve-{version}"):
        return {"choice": "approve", "feedback": feedback.strip()}
    if reject.button("Reject", key=f"reject-{version}"):
        return {"choice": "reject", "feedback": feedback.strip()}
    if edit.button("Edit", key=f"edit-{version}"):
        if not feedback.strip():
            st.warning("Nhập hướng dẫn sửa trước khi chọn Edit.")
            return None
        return {"choice": "edit", "feedback": feedback.strip()}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation · {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("simulation"):
        st.info("Chế độ thử: phiên này sẽ không đăng comment GitHub.")
    if payload.get("risk_factors"):
        st.error("Rủi ro cần chú ý: " + "; ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    answers = {}
    with st.form("escalation_answers"):
        for index, question in enumerate(payload["questions"]):
            st.markdown(f"**{index + 1}. {question}**")
            key = hashlib.sha256(question.encode()).hexdigest()[:12]
            answers[question] = st.text_input("Câu trả lời", key=f"answer-{key}")
        submitted = st.form_submit_button("Gửi câu trả lời")
    if submitted:
        if any(not answer.strip() for answer in answers.values()):
            st.error("Hãy trả lời tất cả câu hỏi trước khi gửi.")
            return None
        return {question: answer.strip() for question, answer in answers.items()}
    return None


async def restore_by_url(thread_id: str):
    async with open_checkpointer(db_path()) as cp:
        graph = build_graph(cp)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        return (snapshot.values or {}).get("pr_url")


with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = asyncio.run(recent_sessions())
        for session in sessions:
            label = f"{session['thread_id'][:8]} · {session['worst_risk']} · {session['pr_url'].rsplit('/', 1)[-1]}"
            st.caption(session["thread_id"])
            st.caption(f"{session['pr_url']} · last event {session['last_event']}")
            if st.button(label, key=f"session-{session['thread_id']}", use_container_width=True):
                restore_thread(session["thread_id"], session["pr_url"])
                st.rerun()
        if not sessions:
            st.caption("Chưa có phiên audit.")
    except Exception as exc:
        st.caption(f"Chưa đọc được audit trail: {exc}")

    if st.button("Cập nhật thống kê confidence", use_container_width=True):
        try:
            st.session_state.confidence_stats = asyncio.run(confidence_statistics())
        except Exception as exc:
            st.session_state.ui_error = f"Không tải được thống kê: {exc}"
    if st.session_state.get("confidence_stats"):
        stats = st.session_state.confidence_stats
        st.subheader(f"{stats['sessions']} phiên đã phân tích")
        st.caption(f"Thời gian xử lý graph trung bình: {stats['average_graph_ms']} ms")
        for branch, count in stats["branches"].items():
            st.write(f"{branch}: {count}")
        st.caption("Lựa chọn cuối theo confidence ban đầu (quan sát, chưa hiệu chỉnh).")
        for bucket, counts in stats["buckets"].items():
            st.write(f"{bucket}: {counts}")

    active_thread = st.session_state.thread_id
    if active_thread:
        with st.expander("Checkpoint history"):
            try:
                async def get_history():
                    async with open_checkpointer(db_path()) as cp:
                        return await checkpoint_history(cp, active_thread)
                history = asyncio.run(get_history())
                for index, item in enumerate(history):
                    confidence = "—" if item["confidence"] is None else f"{item['confidence']:.0%}"
                    st.caption(
                        f"{item['created_at'] or 'checkpoint'} · {item['next']} · "
                        f"confidence {confidence} · review v{item['review_version']}"
                    )
                    if item["has_pending_interrupt"] and "human_approval" in item["next"]:
                        if st.button("Fork mô phỏng trước quyết định", key=f"fork-{index}"):
                            try:
                                async def do_fork():
                                    async with open_checkpointer(db_path()) as cp:
                                        return await fork_approval_checkpoint(
                                            cp, active_thread, item["config"],
                                        )
                                child_id, fork_result = asyncio.run(do_fork())
                                st.session_state.thread_id = child_id
                                st.session_state.pr_url = fork_result.get("pr_url", st.session_state.pr_url)
                                st.session_state.loaded_thread = None
                                receive_result(fork_result)
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Không tạo được phiên mô phỏng: {exc}")
            except Exception as exc:
                st.caption(f"Không đọc được checkpoint history: {exc}")


query_thread = st.query_params.get("thread_id")
if query_thread and query_thread != st.session_state.loaded_thread:
    try:
        pr_for_thread = asyncio.run(restore_by_url(query_thread))
        if pr_for_thread:
            restore_thread(query_thread, pr_for_thread)
    except Exception as exc:
        st.session_state.ui_error = f"Không thể tải phiên từ URL: {exc}"


st.subheader("PR mẫu của lab")
simulation = st.checkbox("Chạy thử, không đăng comment GitHub", value=True)
st.caption("Tắt chế độ thử để chạy thật. Nhánh auto có thể đăng comment ngay sau khi phân tích.")
sample_columns = st.columns(2)
selected_sample = None
for number, column in zip(SAMPLE_PRS, sample_columns):
    with column:
        if st.button(f"Review PR #{number}", key=f"sample-pr-{number}", use_container_width=True):
            selected_sample = SAMPLE_PRS[number]
        st.markdown(f"[Xem PR #{number} trên GitHub]({SAMPLE_PRS[number]})")

with st.form("start_review"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")

if submitted or selected_sample:
    try:
        selected_pr_url = selected_sample or pr_url
        parse_pr_url(selected_pr_url)
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id
        st.session_state.pr_url = selected_pr_url
        st.query_params["thread_id"] = thread_id
        st.session_state.loaded_thread = None
        st.session_state.final = None
        st.session_state.interrupt_payload = None
        with st.spinner("Đang đọc PR và phân tích..."):
            receive_result(asyncio.run(run_graph(selected_pr_url, thread_id, simulation=simulation)))
        if selected_sample:
            st.rerun()
    except Exception as exc:
        st.session_state.ui_error = f"Không thể bắt đầu review: {exc}"

if st.session_state.ui_error:
    st.error(st.session_state.ui_error)

payload = st.session_state.interrupt_payload
if payload is not None:
    answer = render_approval_card(payload) if payload["kind"] == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        try:
            with st.spinner("Đang tiếp tục graph..."):
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id,
                    resume_value=answer,
                ))
            receive_result(result)
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể tiếp tục phiên: {exc}")

final = st.session_state.final
if final is not None:
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if action in {"committed", "auto_committed"}:
        st.success("Review đã được đăng lên GitHub.")
        if analysis:
            st.markdown(f"**Confidence:** {analysis.confidence:.0%} · **Summary:** {analysis.summary}")
        if final.get("posted_comment_url"):
            st.link_button("Mở comment trên GitHub", final["posted_comment_url"])
        elif st.session_state.pr_url:
            st.link_button("Mở PR trên GitHub", st.session_state.pr_url)
    elif action in {"simulation_approved", "auto_simulation_approved"}:
        st.info("Phiên mô phỏng được duyệt; không có comment nào được đăng.")
        if analysis:
            st.markdown(analysis.summary)
    elif action in {"commit_failed", "auto_commit_failed"}:
        st.error("Không đăng được comment. Phiên audit được giữ lại; kiểm tra quyền token hoặc trạng thái kết nối trước khi thử lại.")
        if final.get("commit_error"):
            st.code(final["commit_error"])
        if st.button("Thử đăng lại review đã duyệt", type="primary"):
            try:
                with st.spinner("Đang kiểm tra comment hiện có rồi thử lại..."):
                    receive_result(asyncio.run(retry_graph(st.session_state.thread_id)))
                st.rerun()
            except Exception as exc:
                st.error(f"Không thể thử đăng lại: {exc}")
    elif action == "rejected":
        st.warning("Review bị từ chối; không đăng comment.")
    else:
        st.info(f"Kết quả phiên: {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}")
    if st.session_state.thread_id:
        st.code(f"uv run python -m audit.replay --thread {st.session_state.thread_id}")
