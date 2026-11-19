"""Backend cho giao diện chat: tạo agent, gọi agent, lưu thread ra đĩa.

Không phụ thuộc Streamlit, để logic có thể kiểm tra riêng.

- Agent tạo một lần (lazy) và dùng lại cho mọi thread cùng loại.
- Mọi coroutine chạy trên MỘT event loop nền cố định. NeMo Guardrails và các
  client async gắn với loop đã tạo ra chúng; ``asyncio.run`` mỗi lượt sẽ tạo
  loop mới và có thể làm hỏng các client đó.
- Thread lưu thành ``chat_threads/<id>.json`` ở gốc repo (giữ ngoài git: câu
  trả lời của Red chứa secret demo).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

THREADS_DIR = ROOT / "chat_threads"
NEW_TITLE = "Thread mới"

AGENT_LABELS = {"blue": "Blue", "red": "Red", "red_advance": "Red Advance"}
AGENT_INFO = {
    "blue": "Agent phòng thủ của bạn: rate limit → input guardrail → LLM → output guardrail.",
    "red": "Mục tiêu tấn công mềm: không có guardrail, system prompt cho phép chia sẻ credential với staff.",
    "red_advance": "Mục tiêu tấn công cứng: regex L1 → NeMo input L2 → LLM → regex L3 → NeMo output L4.",
}

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================
# Lưu thread
# ============================================================

def _path(thread_id: str) -> Path:
    if not _ID_RE.match(thread_id or ""):
        raise ValueError(f"thread id không hợp lệ: {thread_id!r}")
    return THREADS_DIR / f"{thread_id}.json"


def new_thread(agent: str) -> dict:
    if agent not in AGENT_LABELS:
        raise ValueError(f"agent không hợp lệ: {agent!r}")
    thread = {
        "id": uuid.uuid4().hex[:12],
        "agent": agent,
        "title": NEW_TITLE,
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    save_thread(thread)
    return thread


def save_thread(thread: dict) -> None:
    THREADS_DIR.mkdir(exist_ok=True)
    thread["updated_at"] = _now()
    _path(thread["id"]).write_text(
        json.dumps(thread, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_thread(thread_id: str) -> dict | None:
    try:
        return json.loads(_path(thread_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_threads() -> list[dict]:
    """Thread mới cập nhật nhất lên đầu; bỏ qua file hỏng."""
    threads = []
    for p in THREADS_DIR.glob("*.json"):
        t = load_thread(p.stem)
        if t and t.get("agent") in AGENT_LABELS:
            threads.append(t)
    return sorted(threads, key=lambda t: t.get("updated_at", ""), reverse=True)


def delete_thread(thread_id: str) -> None:
    _path(thread_id).unlink(missing_ok=True)


# ============================================================
# Gọi agent
# ============================================================

async def _chat(agent, runner, history: list[dict], text: str) -> str:
    """Gửi một tin nhắn, kèm lịch sử thread nếu runner hỗ trợ.

    Không có lịch sử, hoặc runner là Google ADK (Red qua Gemini) → dùng đúng
    ``chat_with_agent`` của lab (một lượt, như grader).

    Có lịch sử + OpenAIRunner → làm như ``OpenAIRunner.chat`` nhưng chèn lịch sử
    vào ``messages``. Lưu ý: input guardrail (của Blue và Red Advance) vẫn chỉ
    xét TIN NHẮN MỚI NHẤT. Đó là đúng hành vi của plugin, và cũng là một lỗ hổng
    thật khi chatbot giữ lịch sử.
    """
    from core.openai_runtime import OpenAIRunner
    from core.utils import chat_with_agent

    if not history or not isinstance(runner, OpenAIRunner):
        reply, _ = await chat_with_agent(agent, runner, text)
        return reply

    for hook in runner.input_hooks:
        blocked = hook(text)
        if blocked:
            return blocked
    block_msg = await runner._run_input_plugins(text)
    if block_msg is not None:
        return block_msg

    messages = [{"role": "system", "content": agent.instruction}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": text})
    completion = runner._client().chat.completions.create(
        model=runner.model, messages=messages, temperature=runner.temperature
    )
    reply = (completion.choices[0].message.content or "").strip()
    for hook in runner.output_hooks:
        reply = hook(reply)
    return await runner._run_output_plugins(reply)


class AgentHub:
    """Giữ agent và event loop nền; dùng chung cho mọi phiên Streamlit."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(
            target=self.loop.run_forever, daemon=True, name="agent-loop"
        ).start()
        self._agents: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def _create(self, kind: str) -> tuple:
        if kind == "blue":
            from agents.agent import create_blue_agent
            from assignment.pipeline import build_production_plugins

            plugins = build_production_plugins()
            agent, runner = create_blue_agent(plugins)
            return agent, runner, plugins
        if kind == "red":
            from agents.agent import create_red_agent_default

            agent, runner = create_red_agent_default()
            return agent, runner, None
        if kind == "red_advance":
            from agents.guards_agent import create_red_agent_advance

            agent, runner = create_red_agent_advance()
            return agent, runner, None
        raise ValueError(f"agent không hợp lệ: {kind!r}")

    def _get(self, kind: str) -> tuple:
        with self._lock:
            if kind not in self._agents:
                self._agents[kind] = self._create(kind)
            return self._agents[kind]

    @staticmethod
    def model_label(kind: str) -> str:
        from core.config import blue_provider_label, red_provider_label

        return blue_provider_label() if kind == "blue" else red_provider_label()

    @staticmethod
    def supports_history(kind: str) -> bool:
        from core.config import red_uses_openai_sdk

        return kind == "blue" or red_uses_openai_sdk()

    def send(self, kind: str, history: list[dict], text: str, *, multi_turn: bool) -> dict:
        """Gọi agent (chặn tới khi xong). Trả {"reply": str, "meta": {...}}."""
        agent, runner, plugins = self._get(kind)
        future = asyncio.run_coroutine_threadsafe(
            self._send(kind, agent, runner, plugins, history if multi_turn else [], text),
            self.loop,
        )
        return future.result(timeout=300)

    async def _send(self, kind, agent, runner, plugins, history, text) -> dict:
        from assignment.pipeline import _BLOCKING_LAYERS, _counters
        from attacks.attacks import classify_attack_outcome, response_leaked_secrets

        before = _counters(plugins) if plugins else None
        t0 = time.perf_counter()
        reply = await _chat(agent, runner, history, text)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        if kind == "blue":
            # Cùng cách pipeline CP3 xác định lớp chặn: chênh lệch bộ đếm plugin.
            after = _counters(plugins)
            layer = next((k for k in after if after[k] > before[k]), None)
            meta = {
                "layer": layer,
                "blocked": layer in _BLOCKING_LAYERS,
                "leaked": response_leaked_secrets(reply),
            }
        else:
            outcome = classify_attack_outcome(
                text, reply, target_name="red_default" if kind == "red" else "red_advance"
            )
            meta = {k: outcome[k] for k in ("layer", "blocked", "leaked")}
        meta["latency_ms"] = latency_ms
        meta["with_history"] = bool(history)
        return {"reply": reply, "meta": meta}
