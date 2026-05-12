"""验证 _langsmith_config / _langsmith_enabled 的行为:env 缺失时返回最小 config,
启用时附 run_name/tags/metadata。不依赖真 LangSmith,只测函数纯逻辑。"""

import importlib
import os

import main


def _reload():
    importlib.reload(main)


def test_disabled_when_env_missing(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    _reload()
    assert main._langsmith_enabled() is False
    cfg = main._langsmith_config("run_x", session_id="s1", tags=["t"], recursion_limit=30)
    assert cfg == {"recursion_limit": 30}


def test_disabled_when_flag_off_even_with_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_xxx")
    _reload()
    assert main._langsmith_enabled() is False
    cfg = main._langsmith_config("run_x")
    assert "run_name" not in cfg


def test_disabled_when_key_missing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    _reload()
    assert main._langsmith_enabled() is False
    cfg = main._langsmith_config("run_x")
    assert "run_name" not in cfg


def test_enabled_emits_trace_metadata(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_test_key")
    _reload()
    assert main._langsmith_enabled() is True
    cfg = main._langsmith_config(
        "run_x",
        session_id="sess-42",
        tags=["a", "b"],
        extra_metadata={"kb_id": "kb-1"},
        recursion_limit=50,
    )
    assert cfg["recursion_limit"] == 50
    assert cfg["run_name"] == "run_x"
    assert cfg["tags"] == ["a", "b"]
    assert cfg["metadata"] == {"session_id": "sess-42", "kb_id": "kb-1"}


def test_enabled_with_minimal_args(monkeypatch):
    """启用时,只传 run_name 也应该正常返回(metadata 为空 dict 时不应注入空 metadata)"""
    monkeypatch.setenv("LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_test_key")
    _reload()
    cfg = main._langsmith_config("run_y")
    assert cfg["run_name"] == "run_y"
    assert cfg["tags"] == []
    # 没有 session_id 也没有 extra_metadata,不应该有 metadata 字段
    assert "metadata" not in cfg


def teardown_module(module):
    """跑完这组测试后,恢复 main 模块默认状态,避免污染其他测试。"""
    os.environ.pop("LANGSMITH_TRACING", None)
    os.environ.pop("LANGSMITH_API_KEY", None)
    importlib.reload(main)
