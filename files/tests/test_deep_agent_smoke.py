"""验证 deepagents 接入: _build_deep_agent / _summarize_deep_run / run_autonomous_chat /
_call_sub_agent 用 fake LLM 跑通,不依赖真 API。"""

import asyncio
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import main


class _BoundFakeModel(GenericFakeChatModel):
    """覆盖 bind_tools / with_structured_output 让 deep agent 内部链路不报 NotImplementedError。"""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return self


def _fake(*responses):
    return _BoundFakeModel(messages=iter(list(responses)))


def test_build_deep_agent_returns_compiled_graph():
    fake = _fake(AIMessage(content="ok"))
    agent = main._build_deep_agent.__wrapped__ if hasattr(main._build_deep_agent, "__wrapped__") else None
    # 不真的 build(会 import 真 ChatOpenAI),改 mock _build_chat_model 路径
    with patch("main._build_chat_model", return_value=fake):
        graph = main._build_deep_agent("k", "u", "m", list(main.TOOLS), "sys")
    assert type(graph).__name__ == "CompiledStateGraph"


def test_summarize_deep_run_extracts_fields():
    fake_msg = AIMessage(content="final answer")
    result = {
        "messages": [fake_msg],
        "todos": [{"id": "1", "content": "todo1", "status": "completed"}],
        "files": {"plan.md": "# plan"},
    }
    summary = main._summarize_deep_run(result)
    assert summary["final"] == "final answer"
    assert len(summary["todos"]) == 1
    assert summary["file_keys"] == ["plan.md"]
    assert isinstance(summary["tool_steps"], list)


def test_summarize_deep_run_handles_missing_fields():
    """todos/files 不存在时返回空,不应报错。"""
    result = {"messages": [AIMessage(content="hi")]}
    summary = main._summarize_deep_run(result)
    assert summary["final"] == "hi"
    assert summary["todos"] == []
    assert summary["file_keys"] == []


def test_summarize_deep_run_handles_empty_dict():
    summary = main._summarize_deep_run({})
    assert summary["final"] == ""
    assert summary["todos"] == []
    assert summary["file_keys"] == []


def test_run_autonomous_chat_uses_deep_agent():
    """end-to-end:run_autonomous_chat 调用 deep agent,fake 模型立即返回 final 文本。"""
    fake = _fake(AIMessage(content="autonomous done"))
    ag = {
        "id": "ag1",
        "name": "auto1",
        "model_name": "fake",
        "available_tools": [],
        "max_steps": 5,
        "system_prompt": "你是测试 agent",
    }
    with patch("main._build_chat_model", return_value=fake):
        out = asyncio.run(main.run_autonomous_chat(ag, "hi", "sess_auto", "k", "u"))
    assert out["response"] == "autonomous done"
    assert out["session_id"] == "sess_auto"
    # autonomous_log 应该有 deep_agent_run 阶段
    assert any(entry.get("phase") == "deep_agent_run" for entry in out["autonomous_log"])
    # 会话历史落了 user + assistant 两条
    history = main.session_histories.get("sess_auto", [])
    assert history[-2:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "autonomous done"},
    ]


def test_run_autonomous_chat_handles_exception():
    """deep agent 抛异常时,应返回友好错误而不是 500。"""
    def boom(*a, **kw):
        raise RuntimeError("simulated failure")
    ag = {
        "id": "ag_err",
        "name": "err",
        "model_name": "fake",
        "available_tools": [],
        "max_steps": 5,
        "system_prompt": "test",
    }
    with patch("main._build_deep_agent", side_effect=boom):
        out = asyncio.run(main.run_autonomous_chat(ag, "hi", "sess_err", "k", "u"))
    assert "失败" in out["response"]
    assert out["autonomous_log"][0]["phase"] == "error"


def test_call_sub_agent_uses_deep_agent():
    """_call_sub_agent 内部应该用 deep agent 而不是 run_agent。"""
    fake = _fake(AIMessage(content="sub said hi"))
    sub = {
        "id": "sub1",
        "name": "sub_a",
        "model_name": "fake",
        "system_prompt": "你是子代理",
    }
    with patch("main._build_chat_model", return_value=fake):
        text, steps = main._call_sub_agent(sub, "ping", "sess_sub", "k", "u")
    assert text == "sub said hi"
    assert isinstance(steps, list)


def test_call_sub_agent_returns_error_on_exception():
    sub = {"id": "x", "name": "x", "model_name": "fake", "system_prompt": "s"}
    with patch("main._build_deep_agent", side_effect=ValueError("nope")):
        text, steps = main._call_sub_agent(sub, "q", "sess_e", "k", "u")
    assert "失败" in text
    assert steps == []
