"""tools 包:所有 @tool 装饰的 LangChain 工具集中注册。

参考 bytedance/deer-flow 的 skills/ 拆分思路:每个工具一个 .py 文件,
顶层 __init__ 把它们聚合成 ALL_TOOLS 列表,main.py 只需 `from tools import ALL_TOOLS`。

加新工具:
1. 新建 tools/<name>.py,导出 @tool 装饰的函数
2. 在本文件 import 并加进 ALL_TOOLS
3. 加测试覆盖

依赖共享状态(知识库 / BGE 模型 / SSL ctx)的工具,
统一从 `runtime` 模块 import,不要 from main import,避免循环依赖。
"""

from .calculator import calculator
from .fetch_url import fetch_url
from .get_current_time import get_current_time
from .get_weather import get_weather
from .search_knowledge_base import search_knowledge_base
from .text_analyzer import text_analyzer
from .unit_converter import unit_converter
from .web_search import web_search
from .word_counter import word_counter

ALL_TOOLS = [
    calculator,
    get_current_time,
    text_analyzer,
    unit_converter,
    word_counter,
    get_weather,
    search_knowledge_base,
    fetch_url,
    web_search,
]

__all__ = [
    "ALL_TOOLS",
    "calculator",
    "get_current_time",
    "text_analyzer",
    "unit_converter",
    "word_counter",
    "get_weather",
    "search_knowledge_base",
    "fetch_url",
    "web_search",
]
