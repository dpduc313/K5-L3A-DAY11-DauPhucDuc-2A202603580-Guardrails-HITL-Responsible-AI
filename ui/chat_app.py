"""Giao diện chat cho Blue, Red, Red Advance. Lịch sử lưu theo thread.

Chạy từ gốc repo (đã kích hoạt .venv):

    streamlit run ui/chat_app.py
"""
from __future__ import annotations

import streamlit as st

from chat_backend import (
    AGENT_INFO,
    AGENT_LABELS,
    NEW_TITLE,
    AgentHub,
    delete_thread,
    list_threads,
    load_thread,
    new_thread,
    save_thread,
)

st.set_page_config(page_title="VinBank agents chat", page_icon=":material/forum:")


@st.cache_resource
def get_hub() -> AgentHub:
    return AgentHub()


def meta_badges(meta: dict | None) -> str:
    """Nhãn dưới mỗi câu trả lời: leak, lớp chặn, latency, có gửi lịch sử không."""
    if not meta:
        return ""
    parts = []
    if meta.get("leaked"):
        parts.append(":red-badge[:material/key: LEAKED]")
    if meta.get("layer") == "error":
        parts.append(":red-badge[:material/error: lỗi]")
    elif meta.get("blocked"):
        parts.append(f":orange-badge[:material/block: chặn: {meta.get('layer')}]")
    elif meta.get("layer"):
        parts.append(f":blue-badge[{meta['layer']}]")
    else:
        parts.append(":green-badge[:material/check: cho qua]")
    if meta.get("latency_ms") is not None:
        parts.append(f":gray-badge[{meta['latency_ms']} ms]")
    if meta.get("with_history"):
        parts.append(":gray-badge[kèm lịch sử]")
    return " ".join(parts)


def start_thread() -> None:
    thread = new_thread(st.session_state.new_agent)
    st.session_state.thread_id = thread["id"]


def open_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id


def remove_thread(thread_id: str) -> None:
    delete_thread(thread_id)
    st.session_state.thread_id = None


hub = get_hub()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# ---------------- Sidebar: tạo và chọn thread ----------------
with st.sidebar:
    st.header("Threads")
    st.segmented_control(
        "Agent cho thread mới",
        options=list(AGENT_LABELS),
        format_func=AGENT_LABELS.get,
        default="blue",
        required=True,
        key="new_agent",
    )
    st.button(
        "Thread mới", icon=":material/add:", on_click=start_thread,
        width="stretch", key="new_thread",
    )
    st.toggle(
        "Gửi kèm lịch sử thread cho model",
        value=True,
        key="multi_turn",
        help=(
            "Bật: model thấy các lượt trước trong thread (chat nhiều lượt thật). "
            "Tắt: mỗi tin nhắn gửi một mình, giống cách lab và grader chạy. "
            "Guardrail input luôn chỉ xét tin nhắn mới nhất."
        ),
    )
    st.divider()
    threads = list_threads()
    if not threads:
        st.caption("Chưa có thread nào.")
    for t in threads:
        st.button(
            f"{AGENT_LABELS[t['agent']]} · {t['title']}",
            key=f"open_{t['id']}",
            on_click=open_thread,
            args=(t["id"],),
            type="primary" if t["id"] == st.session_state.thread_id else "secondary",
            width="stretch",
        )

# ---------------- Main: nội dung thread ----------------
thread = load_thread(st.session_state.thread_id) if st.session_state.thread_id else None
if thread is None:
    st.title("VinBank agents chat")
    st.info(
        "Chọn agent ở thanh bên rồi bấm **Thread mới**, hoặc mở một thread cũ.",
        icon=":material/forum:",
    )
    st.stop()

kind = thread["agent"]
with st.container(horizontal=True, vertical_alignment="bottom"):
    st.title(AGENT_LABELS[kind])
    st.button(
        "Xoá thread", icon=":material/delete:", key="delete_thread",
        on_click=remove_thread, args=(thread["id"],),
    )
st.caption(f"{AGENT_INFO[kind]} Model: `{hub.model_label(kind)}`.")
if st.session_state.multi_turn and not hub.supports_history(kind):
    st.warning("Provider Red hiện tại (Gemini/ADK) chỉ hỗ trợ gửi một lượt; lịch sử chỉ để hiển thị.")

for msg in thread["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.markdown(meta_badges(msg.get("meta")))

if prompt := st.chat_input(f"Nhắn cho {AGENT_LABELS[kind]}", submit_mode="disable"):
    history = [{"role": m["role"], "content": m["content"]} for m in thread["messages"]]
    thread["messages"].append({"role": "user", "content": prompt})
    if thread["title"] == NEW_TITLE:
        thread["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")
    save_thread(thread)

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang chờ agent trả lời…"):
            try:
                result = hub.send(kind, history, prompt, multi_turn=st.session_state.multi_turn)
            except Exception as e:  # lỗi API / NeMo: hiện ra, vẫn lưu vào thread
                result = {
                    "reply": f"Lỗi khi gọi agent: {type(e).__name__}: {e}",
                    "meta": {"layer": "error", "blocked": False, "leaked": False},
                }
        st.markdown(result["reply"])
        st.markdown(meta_badges(result["meta"]))

    thread["messages"].append(
        {"role": "assistant", "content": result["reply"], "meta": result["meta"]}
    )
    save_thread(thread)
    st.rerun()  # cập nhật tiêu đề và thứ tự thread ở sidebar
