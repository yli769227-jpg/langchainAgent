"""验证 run_agent 的 LangGraph 调度链路：mock LLM 返回 tool_call → agent 执行
真实工具 → 第二轮返回 final text → run_agent 应当返回正确的 (text, steps) 元组。

不依赖任何真 API。"""

from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import main


class _BoundFakeModel(GenericFakeChatModel):
    """GenericFakeChatModel 默认没实现 bind_tools，create_agent 调 bind_tools 会
    NotImplementedError。这里覆盖为 self（fake 不真绑定，按 messages 队列回放）。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_with(*responses):
    return _BoundFakeModel(messages=iter(list(responses)))


def test_run_agent_single_tool_call():
    fake = _fake_with(
        AIMessage(
            content="",
            tool_calls=[{
                "name": "calculator",
                "args": {"expression": "1+1"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="答案是 2"),
    )

    with patch.object(main, "_build_chat_model", return_value=fake):
        text, steps = main.run_agent(
            "sk-test", "http://localhost:9999/v1", "fake-model",
            [{"role": "user", "content": "1+1=?"}],
        )

    assert text == "答案是 2"
    assert len(steps) == 1
    step = steps[0]
    assert step["tool"] == "calculator"
    assert step["input"] == {"expression": "1+1"}
    assert "= 2" in step["output"]


def test_run_agent_no_tool_call_direct_answer():
    fake = _fake_with(AIMessage(content="你好"))

    with patch.object(main, "_build_chat_model", return_value=fake):
        text, steps = main.run_agent(
            "sk", "http://x", "m",
            [{"role": "user", "content": "hi"}],
        )

    assert text == "你好"
    assert steps == []


def test_run_agent_multi_step_chain():
    """两次 tool_call: 先 calculator 算 2+3=5, 再 word_counter 数 'a'。"""
    fake = _fake_with(
        AIMessage(
            content="",
            tool_calls=[{"name": "calculator", "args": {"expression": "2+3"}, "id": "c1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "word_counter", "args": {"text": "abc abc", "target_word": "a"}, "id": "c2", "type": "tool_call"}],
        ),
        AIMessage(content="完成"),
    )

    with patch.object(main, "_build_chat_model", return_value=fake):
        text, steps = main.run_agent("sk", "http://x", "m", [{"role": "user", "content": "go"}])

    assert text == "完成"
    assert len(steps) == 2
    assert steps[0]["tool"] == "calculator"
    assert steps[1]["tool"] == "word_counter"
    # 工具结果应当被正确关联
    assert "= 5" in steps[0]["output"]
    assert "2" in steps[1]["output"]


def test_run_agent_handles_invoke_exception():
    """create_agent.invoke 抛异常时不应崩溃，返回错误消息和空 steps。"""
    class _ExplodingModel(_BoundFakeModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("simulated network failure")

    fake = _ExplodingModel(messages=iter([AIMessage(content="x")]))
    with patch.object(main, "_build_chat_model", return_value=fake):
        text, steps = main.run_agent("sk", "http://x", "m", [{"role": "user", "content": "hi"}])

    assert "处理过程中遇到了问题" in text
    assert steps == []


def test_extract_steps_helper_pairs_tool_calls():
    """_extract_steps_from_messages 应当把 AIMessage.tool_calls 与 ToolMessage 通过 id 配对。"""
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "abc", "type": "tool_call"}],
        ),
        ToolMessage(content="计算结果: 1+1 = 2", tool_call_id="abc", name="calculator"),
        AIMessage(content="2"),
    ]

    steps = main._extract_steps_from_messages(messages)
    assert len(steps) == 1
    assert steps[0]["tool"] == "calculator"
    assert steps[0]["input"] == {"expression": "1+1"}
    assert "= 2" in steps[0]["output"]
