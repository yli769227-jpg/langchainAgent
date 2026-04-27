"""AST 白名单计算器的安全性测试。

calculator 内部用 _calc_eval：
- 只放行数字常量、命名查找（仅白名单 _CALC_NAMES）、双目/一元算术、函数调用
- 任何 import / lambda / 属性访问 / 推导式 / subscript 都应当被拒绝
- 嵌套深度上限 50（超过即抛 ValueError）
"""

import pytest

import main


def _eval(expr: str):
    """直接走 _calc_eval，便于断言异常类型；calculator 包了一层 try/except 把异常变字符串。"""
    import ast
    tree = ast.parse(expr, mode="eval")
    return main._calc_eval(tree)


# ---------- 白名单允许 ----------

def test_allow_basic_arith():
    assert _eval("1 + 2 * 3") == 7


def test_allow_unary_minus():
    assert _eval("-5 + 3") == -2


def test_allow_pow():
    assert _eval("2 ** 10") == 1024


def test_allow_whitelisted_func_sqrt():
    assert _eval("sqrt(16)") == 4.0


def test_allow_whitelisted_const_pi():
    import math
    assert _eval("pi") == math.pi


def test_allow_abs_min_max():
    assert _eval("abs(-7)") == 7
    assert _eval("min(1, 2, 3)") == 1
    assert _eval("max(1, 2, 3)") == 3


# ---------- 黑名单拒绝 ----------

def test_reject_import():
    with pytest.raises(ValueError):
        _eval("__import__('os')")


def test_reject_lambda():
    with pytest.raises(ValueError):
        _eval("(lambda x: x)(1)")


def test_reject_attribute_access():
    """属性访问 (e.g. ().__class__) 必须拒绝。"""
    with pytest.raises(ValueError):
        _eval("(1).__class__")


def test_reject_subscript():
    with pytest.raises(ValueError):
        _eval("[1,2,3][0]")


def test_reject_unknown_name():
    with pytest.raises(ValueError, match="未知标识符"):
        _eval("evil_func")


def test_reject_unknown_func_call():
    with pytest.raises(ValueError):
        _eval("eval('1+1')")


def test_reject_list_comprehension():
    with pytest.raises(ValueError):
        _eval("[x*x for x in range(10)]")


def test_reject_string_literal():
    """白名单只放行 (int, float)，字符串应当拒绝。"""
    with pytest.raises(ValueError):
        _eval("'hello'")


# ---------- 深度限制 ----------

def test_allow_moderate_nesting():
    # 5 层嵌套，远低于 depth 上限 (50)
    expr = "(((1 + 2) * 3) - (4 / 2))"
    assert _eval(expr) == 7.0


def test_reject_excessive_nesting():
    """构造一个深度远超 50 的表达式，必须抛 ValueError('表达式嵌套过深')。"""
    expr = "1" + ("+1" * 100)  # 100 层 BinOp 嵌套
    with pytest.raises(ValueError, match="嵌套过深"):
        _eval(expr)


# ---------- calculator wrapper 行为 ----------

def test_calculator_wraps_errors_in_text():
    """calculator 工具会把异常包装成"计算错误: ..."字符串而不是抛出。"""
    out = main.calculator.invoke({"expression": "__import__('os')"})
    assert "计算错误" in out


def test_calculator_length_limit():
    out = main.calculator.invoke({"expression": "1" * 600})
    assert "过长" in out
