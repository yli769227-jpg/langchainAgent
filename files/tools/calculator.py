"""calculator 工具:基于 AST 白名单的安全数学计算器。

不走 eval,只放行算术 + 指定数学函数/常量,杜绝任意代码执行。
"""

import ast
import math
import operator

from langchain_core.tools import tool

_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_CALC_NAMES = {
    n: getattr(math, n) for n in (
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "sqrt", "log", "log2", "log10",
        "exp", "ceil", "floor", "factorial", "gcd",
        "degrees", "radians", "pi", "e", "tau", "inf",
    )
}
_CALC_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})


def _calc_eval(node, depth: int = 0):
    """AST 节点白名单求值。depth 防爆栈,只放行 Constant/Name/BinOp/UnaryOp/Call。"""
    if depth > 50:
        raise ValueError("表达式嵌套过深")
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CALC_NAMES:
            return _CALC_NAMES[node.id]
        raise ValueError(f"未知标识符: {node.id}")
    if isinstance(node, ast.BinOp):
        op = _CALC_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
        return op(_calc_eval(node.left, depth + 1), _calc_eval(node.right, depth + 1))
    if isinstance(node, ast.UnaryOp):
        op = _CALC_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不允许的一元运算符")
        return op(_calc_eval(node.operand, depth + 1))
    if isinstance(node, ast.Call):
        fn = _calc_eval(node.func, depth + 1)
        if not callable(fn):
            raise ValueError("非可调用对象")
        args = [_calc_eval(a, depth + 1) for a in node.args]
        return fn(*args)
    raise ValueError(f"不允许的语法: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持基本四则运算、幂运算、三角函数、对数等。示例: '2 + 3 * 4', 'sqrt(16)'。

    Args:
        expression: 数学表达式字符串
    """
    if len(expression) > 500:
        return "计算错误: 表达式过长（上限 500 字符）"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _calc_eval(tree)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"
