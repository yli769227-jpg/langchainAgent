"""工具 happy-path 测试。@tool 装饰后，调用约定改为 .invoke(args_dict)。"""

import datetime

import main


def test_calculator_basic_arith():
    out = main.calculator.invoke({"expression": "1+2*3"})
    assert "= 7" in out


def test_calculator_sin_zero():
    out = main.calculator.invoke({"expression": "sin(0)"})
    # sin(0) = 0.0
    assert "= 0" in out


def test_calculator_rejects_unknown_name():
    out = main.calculator.invoke({"expression": "evil_func()"})
    # 应当走 except 分支，返回错误信息
    assert "计算错误" in out
    assert "未知标识符" in out or "evil_func" in out


def test_calculator_too_long():
    out = main.calculator.invoke({"expression": "1+" * 300})
    assert "过长" in out


def test_get_current_time_contains_year():
    out = main.get_current_time.invoke({})
    year = str(datetime.datetime.now().year)
    assert year in out
    assert "星期" in out


def test_text_analyzer_counts():
    out = main.text_analyzer.invoke({"text": "hello world\nfoo bar baz"})
    assert "总字符数" in out
    assert "行数: 2" in out
    # 5 个英文 token: hello / world / foo / bar / baz
    assert "英文单词数: 5" in out


def test_text_analyzer_chinese():
    out = main.text_analyzer.invoke({"text": "你好世界"})
    assert "中文字符数: 4" in out


def test_unit_converter_km_to_mile():
    out = main.unit_converter.invoke({"value": 100, "from_unit": "km", "to_unit": "mile"})
    # 100 km ≈ 62.1371 mile
    assert "62." in out
    assert "mile" in out


def test_unit_converter_celsius_to_fahrenheit():
    out = main.unit_converter.invoke({"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"})
    assert "212" in out  # 100°C = 212°F


def test_unit_converter_unsupported():
    out = main.unit_converter.invoke({"value": 1, "from_unit": "foo", "to_unit": "bar"})
    assert "不支持" in out


def test_word_counter():
    out = main.word_counter.invoke({"text": "Hello hello HELLO world", "target_word": "hello"})
    assert "3" in out


def test_word_counter_zero():
    out = main.word_counter.invoke({"text": "abc def", "target_word": "xyz"})
    assert "0" in out


def test_search_knowledge_base_missing():
    out = main.search_knowledge_base.invoke({"kb_id": "does_not_exist", "query": "anything"})
    assert "不存在" in out


def test_search_knowledge_base_empty():
    main.knowledge_bases["kb_test_empty"] = {"name": "t", "description": "", "docs": []}
    try:
        out = main.search_knowledge_base.invoke({"kb_id": "kb_test_empty", "query": "x"})
        assert "为空" in out
    finally:
        del main.knowledge_bases["kb_test_empty"]


def test_search_knowledge_base_keyword_match():
    """没有 BGE 模型时降级为关键词检索。"""
    kb_id = "kb_test_kw"
    main.knowledge_bases[kb_id] = {
        "name": "t",
        "description": "",
        "docs": ["LangChain 是一个 LLM 框架", "Python 是脚本语言", "今天天气很好"],
    }
    try:
        out = main.search_knowledge_base.invoke({"kb_id": kb_id, "query": "LangChain"})
        # 至少应该命中第一条 doc
        assert "LangChain" in out
    finally:
        del main.knowledge_bases[kb_id]


def test_tools_schema_derived():
    """TOOLS_SCHEMA 应当从 @tool 函数自动派生，长度等于 TOOLS。"""
    assert len(main.TOOLS_SCHEMA) == len(main.TOOLS)
    names = {s["function"]["name"] for s in main.TOOLS_SCHEMA}
    assert names == {t.name for t in main.TOOLS}


def test_tool_functions_dict_dispatch():
    """TOOL_FUNCTIONS dict 应当可以用 .invoke 调度。"""
    out = main.TOOL_FUNCTIONS["calculator"].invoke({"expression": "2+2"})
    assert "= 4" in out
