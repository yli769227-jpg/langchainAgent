"""
LangChain Agent Backend - FastAPI

真接入 LangChain 1.2 + LangGraph 1.1 栈：
- 工具用 @tool 装饰，schema 自动派生
- run_agent 走 langchain.agents.create_agent (LangGraph 状态机)
- /chat/stream 用 agent.astream_events 流式
- 兼容第三方/中转 OpenAI 格式接口（任意 base_url）
"""

import os
import ast
import math
import operator
import datetime
import json
import logging
import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import urllib.request
import urllib.parse
import ssl

# LangChain / LangGraph 栈
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent, SubAgent

# 在任何 langchain/langsmith import 触发 env 读取前先加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("agent")


def _langsmith_enabled() -> bool:
    """LANGSMITH_TRACING=true 且配了 API key 才视作启用。缺任一即静默不上报。"""
    flag = os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    return flag and has_key


def _langsmith_config(
    run_name: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict | None = None,
    recursion_limit: int = 25,
) -> dict:
    """构造 LangChain/LangGraph invoke 的 config。

    缺 LANGSMITH_TRACING 或 LANGSMITH_API_KEY 时只返回最小 config(不带 trace 元数据),
    保证未配 LangSmith 的环境下行为不变。
    """
    cfg: dict = {"recursion_limit": recursion_limit}
    if not _langsmith_enabled():
        return cfg
    cfg["run_name"] = run_name
    cfg["tags"] = list(tags or [])
    metadata: dict = {}
    if session_id:
        metadata["session_id"] = session_id
    if extra_metadata:
        metadata.update(extra_metadata)
    if metadata:
        cfg["metadata"] = metadata
    return cfg


# 启动时单次状态日志,便于排查 trace 是否真的启用
if _langsmith_enabled():
    logger.info(
        "[langsmith] tracing enabled project=%s endpoint=%s",
        os.getenv("LANGSMITH_PROJECT", "<default>"),
        os.getenv("LANGSMITH_ENDPOINT", "<default>"),
    )
else:
    logger.info("[langsmith] tracing disabled (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to enable)")


def mask_key(k: str) -> str:
    """掩码 API Key，只保留前 6 后 4 便于排查又不泄漏。"""
    if not k:
        return "<empty>"
    if len(k) <= 10:
        return "***"
    return f"{k[:6]}...{k[-4:]}"


# 第三方中转服务可能用自签证书，默认仍校验；如需关闭由环境变量开启
_ssl_ctx = ssl.create_default_context()
if os.getenv("AGENT_INSECURE_SSL") == "1":
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE
    logger.warning("SSL 证书验证已关闭 (AGENT_INSECURE_SSL=1)")

app = FastAPI(title="LangChain Agent API", version="2.0.0")

# CORS 白名单：允许本地开发 + file:// 打开前端
_allowed_origin_re = re.compile(r"^(https?://localhost(:\d+)?|https?://127\.0\.0\.1(:\d+)?|null)$")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_allowed_origin_re.pattern,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Tools 实现 ====================

# AST 白名单计算器：不走 eval，只放行算术 + 指定函数/常量
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


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前日期和时间（北京时间）。

    Args:
        timezone: 时区，默认 Asia/Shanghai
    """
    now = datetime.datetime.now()
    return (
        f"当前时间（北京时间）: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期: {['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}\n"
        f"今年第 {now.timetuple().tm_yday} 天"
    )


@tool
def text_analyzer(text: str) -> str:
    """分析文本统计信息：字符数、词数、行数、中文字符数等。

    Args:
        text: 要分析的文本
    """
    lines = text.split('\n')
    words = text.split()
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return (
        f"📊 文本分析结果:\n"
        f"  - 总字符数: {len(text)}\n"
        f"  - 中文字符数: {chinese_chars}\n"
        f"  - 英文单词数: {len(words)}\n"
        f"  - 行数: {len(lines)}\n"
        f"  - 段落数: {len([l for l in lines if l.strip()])}"
    )


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。支持长度(m,km,mile,ft,cm,mm)、重量(kg,lb,g,oz)、温度(celsius,fahrenheit,kelvin)。

    Args:
        value: 要换算的数值
        from_unit: 原单位
        to_unit: 目标单位
    """
    conversions = {
        ("m", "km"): 0.001,         ("km", "m"): 1000,
        ("m", "mile"): 0.000621371, ("mile", "m"): 1609.34,
        ("m", "ft"): 3.28084,       ("ft", "m"): 0.3048,
        ("m", "cm"): 100,           ("cm", "m"): 0.01,
        ("m", "mm"): 1000,          ("mm", "m"): 0.001,
        ("km", "mile"): 0.621371,   ("mile", "km"): 1.60934,
        ("cm", "mm"): 10,           ("mm", "cm"): 0.1,
        ("kg", "lb"): 2.20462,      ("lb", "kg"): 0.453592,
        ("kg", "g"): 1000,          ("g", "kg"): 0.001,
        ("kg", "oz"): 35.274,       ("oz", "kg"): 0.0283495,
        ("g", "oz"): 0.035274,      ("oz", "g"): 28.3495,
    }
    key = (from_unit.lower(), to_unit.lower())
    f, t = from_unit.lower(), to_unit.lower()
    if f == "celsius" and t == "fahrenheit":
        result = value * 9/5 + 32
    elif f == "fahrenheit" and t == "celsius":
        result = (value - 32) * 5/9
    elif f == "celsius" and t == "kelvin":
        result = value + 273.15
    elif f == "kelvin" and t == "celsius":
        result = value - 273.15
    elif key in conversions:
        result = value * conversions[key]
    else:
        return f"不支持 {from_unit} 到 {to_unit} 的换算"
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


@tool
def word_counter(text: str, target_word: str) -> str:
    """在文本中统计特定词语出现的次数（不区分大小写）。

    Args:
        text: 要搜索的文本
        target_word: 要统计的词语
    """
    count = text.lower().count(target_word.lower())
    return f"词语 '{target_word}' 在文本中出现了 {count} 次"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，包括温度、湿度、风速、天气状况等。

    Args:
        city: 城市名称，支持中文或英文，如 '北京'、'Shanghai'、'London'
    """
    try:
        # 第一步：用 open-meteo geocoding API 将城市名转为经纬度
        encoded_city = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=zh&format=json"
        req = urllib.request.Request(geo_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            geo_data = json.loads(resp.read().decode())

        if not geo_data.get("results"):
            return f"找不到城市: {city}"

        result = geo_data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        city_name = result.get("name", city)
        country = result.get("country", "")

        # 第二步：用经纬度查询实时天气
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"weather_code,wind_speed_10m,wind_direction_10m,visibility"
            f"&wind_speed_unit=kmh&timezone=auto"
        )
        req2 = urllib.request.Request(weather_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=10, context=_ssl_ctx) as resp2:
            w_data = json.loads(resp2.read().decode())

        cur = w_data["current"]
        temp = cur["temperature_2m"]
        feels_like = cur["apparent_temperature"]
        humidity = cur["relative_humidity_2m"]
        wind_speed = cur["wind_speed_10m"]
        wind_dir = cur["wind_direction_10m"]
        visibility = cur.get("visibility", "N/A")
        wmo_code = cur["weather_code"]

        # WMO 天气代码简单映射
        wmo_desc = {
            0: "晴天", 1: "基本晴朗", 2: "局部多云", 3: "阴天",
            45: "雾", 48: "冻雾",
            51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
            95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
        }
        desc = wmo_desc.get(wmo_code, f"天气代码 {wmo_code}")

        vis_str = f"{int(visibility/1000)} km" if isinstance(visibility, (int, float)) else str(visibility)

        return (
            f"🌤 {city_name}, {country} 实时天气\n"
            f"  天气状况: {desc}\n"
            f"  温度: {temp}°C（体感 {feels_like}°C）\n"
            f"  湿度: {humidity}%\n"
            f"  风速: {wind_speed} km/h，风向: {wind_dir}°\n"
            f"  能见度: {vis_str}"
        )
    except Exception as e:
        return f"天气查询失败: {str(e)}"


# ==================== 知识库 ====================

import uuid

# 内存存储：{kb_id: {"name": str, "description": str, "docs": [str], "vectors": list}}
knowledge_bases: dict[str, dict] = {}

# BGE 模型懒加载
_bge_model = None

def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _bge_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except Exception:
            _bge_model = False  # 标记加载失败，降级为纯关键词
    return _bge_model if _bge_model is not False else None


def _keyword_score(doc: str, query: str) -> float:
    query_words = set(query.lower().split())
    doc_lower = doc.lower()
    score = sum(1 for w in query_words if w in doc_lower)
    score += sum(1 for c in query if c.strip() and c in doc)
    return float(score)


def _encode(texts: list) -> list:
    model = _get_bge_model()
    if model is None:
        return []
    # BGE 建议查询加前缀
    return model.encode(texts, normalize_embeddings=True).tolist()


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # 已 normalize，dot == cosine


@tool
def search_knowledge_base(kb_id: str, query: str, top_k: int = 3) -> str:
    """在指定知识库中搜索相关内容，用于回答基于知识库的问题。

    Args:
        kb_id: 知识库 ID
        query: 搜索查询内容
        top_k: 返回最相关的文档数量，默认 3
    """
    if kb_id not in knowledge_bases:
        return f"知识库 {kb_id} 不存在"
    kb = knowledge_bases[kb_id]
    docs = kb["docs"]
    if not docs:
        return "知识库为空"

    vectors = kb.get("vectors", [])
    model = _get_bge_model()

    scored = []
    if model and len(vectors) == len(docs):
        # 向量检索
        query_vec = _encode(["为这个句子生成表示以用于检索相关文章：" + query])[0]
        for i, doc in enumerate(docs):
            vec_score = _cosine(query_vec, vectors[i])
            kw_score = _keyword_score(doc, query)
            # 归一化关键词分数后融合，向量权重 0.7，关键词权重 0.3
            max_kw = max((_keyword_score(d, query) for d in docs), default=1) or 1
            combined = 0.7 * vec_score + 0.3 * (kw_score / max_kw)
            scored.append((combined, doc))
    else:
        # 降级：纯关键词
        for doc in docs:
            scored.append((_keyword_score(doc, query), doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [doc for score, doc in scored[:top_k] if score > 0]
    if not results:
        return "未找到相关内容"
    return "\n\n---\n\n".join(results)


@tool
def fetch_url(url: str, max_chars: int = 2000) -> str:
    """抓取任意 HTTP(S) URL 的正文内容。HTML 会自动剥标签并压缩空白；JSON/纯文本原样返回。超过 max_chars 会截断。用于读文章/API/在线文档。

    Args:
        url: 目标 URL，必须以 http:// 或 https:// 开头
        max_chars: 返回正文最大字符数，默认 2000，上限 8000
    """
    if not re.match(r"^https?://", url):
        return "url 必须以 http:// 或 https:// 开头"
    if max_chars > 8000:
        max_chars = 8000
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AgentFlow)"})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(1024 * 1024)  # 最多 1MB
            body = raw.decode(resp.headers.get_content_charset() or "utf-8", errors="replace")
        # 简易 HTML→文本：去 script/style 再剥标签
        if "html" in ctype.lower() or body.lstrip().startswith("<"):
            body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body, flags=re.DOTALL | re.IGNORECASE)
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
        truncated = len(body) > max_chars
        body = body[:max_chars]
        return f"URL: {url}\nContent-Type: {ctype}\n\n{body}" + ("\n\n[... 已截断]" if truncated else "")
    except Exception as e:
        return f"抓取失败: {type(e).__name__}: {str(e)}"


@tool
def web_search(query: str) -> str:
    """用 DuckDuckGo 搜索网络信息，返回摘要和相关主题。适合快速查概念/时事/知识问答，无需 API Key。若需完整正文请用 fetch_url。

    Args:
        query: 搜索关键词或自然语言问题
    """
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
        req = urllib.request.Request(f"https://api.duckduckgo.com/?{params}", headers={"User-Agent": "Mozilla/5.0 (AgentFlow)"})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        parts = []
        if data.get("AbstractText"):
            parts.append(f"摘要: {data['AbstractText']}")
        if data.get("AbstractURL"):
            parts.append(f"来源: {data['AbstractURL']}")
        related = [r.get("Text", "") for r in data.get("RelatedTopics", []) if isinstance(r, dict) and r.get("Text")]
        if related:
            parts.append("相关结果:\n" + "\n".join(f"- {r}" for r in related[:5]))
        if not parts:
            return f"没有直接答案。建议用 fetch_url 抓取具体网页。查询: {query}"
        return "\n\n".join(parts)
    except Exception as e:
        return f"搜索失败: {type(e).__name__}: {str(e)}"


# ==================== LangChain 工具注册 ====================

# 所有 @tool 装饰的工具集中注册
TOOLS = [
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


def _tools_to_openai_schema(tools: list) -> list:
    """从 @tool 装饰的函数列表派生 OpenAI tools 格式 schema list（兼容旧调用方）。"""
    return [convert_to_openai_tool(t) for t in tools]


# 兼容性导出：原代码引用 TOOLS_SCHEMA / TOOL_FUNCTIONS 的地方继续可用
# 注意：TOOL_FUNCTIONS 的值是 LangChain Tool 对象，调用时必须用 .invoke(args_dict) 而不是 (**kwargs)
TOOLS_SCHEMA = _tools_to_openai_schema(TOOLS)
TOOL_FUNCTIONS = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """你是一个智能助手。请根据用户的问题给出准确、有用的回答。
回答时请用中文，并保持友好、专业的语气。
如果需要使用工具来回答问题，请直接使用，不要向用户列举你的工具能力。"""

# 全局会话历史
session_histories: dict[str, list] = {}


def _build_chat_model(api_key: str, base_url: str, model_name: str, **kwargs):
    """构造 ChatOpenAI 客户端，统一从此入口便于 mock。"""
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, **kwargs)


def _extract_steps_from_messages(messages: list) -> list:
    """从 LangGraph 返回的 messages 列表里提取 [{tool, input, output}] 步骤。

    LangChain 消息流约定：AIMessage 带 tool_calls -> 后续 ToolMessage 用 tool_call_id 关联。
    """
    # 先把所有 ToolMessage 按 tool_call_id 索引
    tool_results = {}
    for m in messages:
        if type(m).__name__ == "ToolMessage":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                tool_results[tcid] = str(getattr(m, "content", ""))

    steps = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            # tool_calls 是 dict 列表，形如 {'name':..., 'args':..., 'id':...}
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tcid = tc.get("id", "")
            else:
                name = getattr(tc, "name", "")
                args = getattr(tc, "args", {}) or {}
                tcid = getattr(tc, "id", "")
            steps.append({
                "tool": name,
                "input": args,
                "output": tool_results.get(tcid, ""),
            })
    return steps


def run_agent(
    api_key: str,
    base_url: str,
    model_name: str,
    messages: list,
    *,
    extra_config: dict | None = None,
) -> tuple[str, list]:
    """用 LangChain create_agent + LangGraph 执行 ReAct 循环。

    返回 (最终回复文本, [{"tool":..., "input":..., "output":...}])。
    extra_config 是可选的 LangChain invoke config(用于注入 LangSmith run_name/tags/metadata),
    省略时行为与旧版完全一致。
    """
    logger.info("[run_agent] 启动 LangGraph agent model=%s base=%s msgs=%d",
                model_name, base_url, len(messages))
    model = _build_chat_model(api_key, base_url, model_name)
    agent = create_agent(model=model, tools=TOOLS)

    invoke_config: dict = {"recursion_limit": 25}
    if extra_config:
        invoke_config.update(extra_config)

    try:
        result = agent.invoke(
            {"messages": messages},
            config=invoke_config,
        )
    except Exception as e:
        logger.exception("[run_agent] LangGraph invoke 异常: %s", e)
        return f"抱歉，处理过程中遇到了问题：{type(e).__name__}: {e}", []

    out_msgs = result.get("messages", [])
    final_text = ""
    if out_msgs:
        last = out_msgs[-1]
        final_text = getattr(last, "content", "") or ""

    steps = _extract_steps_from_messages(out_msgs)
    logger.info("[run_agent] 完成 final_len=%d steps=%d", len(final_text), len(steps))
    return final_text, steps


# ==================== API Models ====================

# 凭证一律从前端请求传入，服务器不保存
class _WithCreds(BaseModel):
    api_key: str
    base_url: str


class ChatRequest(_WithCreds):
    message: str
    session_id: str = "default"
    model_name: str = "claude-sonnet-4-6"
    kb_id: Optional[str] = None


# ==================== API Routes ====================

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Agent is running"}


class LoginRequest(BaseModel):
    password: str


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """后端校验密码，密码从 AGENT_LOGIN_PASSWORD 环境变量读，未设置则用默认值。"""
    expected = os.getenv("AGENT_LOGIN_PASSWORD", "agent2024")
    if req.password == expected:
        logger.info("登录成功")
        return {"ok": True}
    logger.warning("登录失败：密码不匹配")
    raise HTTPException(status_code=401, detail="密码错误")


@app.get("/tools")
async def get_tools():
    return {
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"]
            }
            for t in TOOLS_SCHEMA
        ]
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    logger.info("POST /chat session=%s model=%s key=%s base=%s msg_len=%d",
                req.session_id, req.model_name, mask_key(req.api_key), req.base_url, len(req.message))
    if not req.api_key or not req.base_url:
        raise HTTPException(status_code=400, detail="缺少 api_key 或 base_url（请在前端设置中填写）")
    try:
        history = session_histories.get(req.session_id, [])

        # 如果指定了知识库，在 system prompt 里告知 agent
        system = SYSTEM_PROMPT
        if req.kb_id and req.kb_id in knowledge_bases:
            kb = knowledge_bases[req.kb_id]
            system += f"\n\n当前对话已关联知识库「{kb['name']}」(ID: {req.kb_id})。当用户提问与该知识库相关时，请使用 search_knowledge_base 工具（kb_id=\"{req.kb_id}\"）检索后再回答。"

        messages = (
            [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": req.message}]
        )

        trace_cfg = _langsmith_config(
            run_name="chat",
            session_id=req.session_id,
            tags=["endpoint:chat", f"model:{req.model_name}"],
            extra_metadata={"kb_id": req.kb_id} if req.kb_id else None,
        )
        response_text, steps = await asyncio.to_thread(
            run_agent, req.api_key, req.base_url, req.model_name, messages,
            extra_config=trace_cfg,
        )

        # 更新历史，保留最近 20 条
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": response_text})
        session_histories[req.session_id] = history[-20:]

        return {
            "response": response_text,
            "steps": steps,
            "session_id": req.session_id,
        }

    except Exception as e:
        logger.exception("/chat 处理异常 session=%s", req.session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    logger.info("POST /chat/stream session=%s model=%s key=%s base=%s msg_len=%d",
                req.session_id, req.model_name, mask_key(req.api_key), req.base_url, len(req.message))
    if not req.api_key or not req.base_url:
        raise HTTPException(status_code=400, detail="缺少 api_key 或 base_url（请在前端设置中填写）")
    history = session_histories.get(req.session_id, [])
    system = SYSTEM_PROMPT
    if req.kb_id and req.kb_id in knowledge_bases:
        kb = knowledge_bases[req.kb_id]
        system += f"\n\n当前对话已关联知识库「{kb['name']}」(ID: {req.kb_id})。当用户提问与该知识库相关时，请使用 search_knowledge_base 工具（kb_id=\"{req.kb_id}\"）检索后再回答。"

    messages = (
        [{"role": "system", "content": system}]
        + history
        + [{"role": "user", "content": req.message}]
    )

    async def generate():
        logger.info("[chat/stream] 启动 LangGraph 流式 agent session=%s", req.session_id)
        model = _build_chat_model(req.api_key, req.base_url, req.model_name, streaming=True)
        agent = create_agent(model=model, tools=TOOLS)

        steps: list = []
        final_text_parts: list = []

        stream_cfg = {"recursion_limit": 25}
        stream_cfg.update(_langsmith_config(
            run_name="chat_stream",
            session_id=req.session_id,
            tags=["endpoint:chat_stream", f"model:{req.model_name}"],
            extra_metadata={"kb_id": req.kb_id} if req.kb_id else None,
        ))
        try:
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config=stream_cfg,
            ):
                kind = event.get("event")
                data = event.get("data", {}) or {}

                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    content = getattr(chunk, "content", "") if chunk is not None else ""
                    # content 可能是 str 或 list[dict] (Anthropic 风格)
                    if isinstance(content, list):
                        text = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict) and seg.get("type") == "text")
                    else:
                        text = content or ""
                    if text:
                        final_text_parts.append(text)
                        yield f"data: {json.dumps({'type':'token','content':text}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    tool_input = data.get("input", {}) or {}
                    raw_output = data.get("output")
                    # output 可能是 ToolMessage 对象，取 content
                    tool_output = getattr(raw_output, "content", raw_output)
                    tool_output = str(tool_output) if tool_output is not None else ""
                    step = {"tool": tool_name, "input": tool_input, "output": tool_output}
                    steps.append(step)
                    logger.info("[chat/stream] 工具调用 tool=%s input=%s out_len=%d", tool_name, tool_input, len(tool_output))
                    yield f"data: {json.dumps({'type':'tool','step':step}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("[chat/stream] astream_events 异常: %s", e)
            err = f"流式处理异常：{type(e).__name__}: {e}"
            yield f"data: {json.dumps({'type':'token','content':err}, ensure_ascii=False)}\n\n"

        full = "".join(final_text_parts)
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": full})
        session_histories[req.session_id] = history[-20:]
        logger.info("[chat/stream] 完成 final_len=%d steps=%d", len(full), len(steps))
        yield f"data: {json.dumps({'type':'done','steps':steps}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/chat/history/{session_id}")
async def clear_history(session_id: str):
    """清空指定会话的历史。"""
    if session_id in session_histories:
        del session_histories[session_id]
        logger.info("[clear_history] session=%s 已清空", session_id)
    return {"message": f"Session {session_id} cleared"}


# ==================== 知识库 API ====================

@app.post("/knowledge-base")
async def create_kb(name: str, description: str = ""):
    kb_id = str(uuid.uuid4())[:8]
    knowledge_bases[kb_id] = {"name": name, "description": description, "docs": []}
    return {"kb_id": kb_id, "name": name}


@app.get("/knowledge-base")
async def list_kbs():
    return {
        "knowledge_bases": [
            {"kb_id": kid, "name": v["name"], "description": v["description"], "doc_count": len(v["docs"])}
            for kid, v in knowledge_bases.items()
        ]
    }


class DocumentRequest(BaseModel):
    content: str

@app.post("/knowledge-base/{kb_id}/document")
async def add_document(kb_id: str, req: DocumentRequest):
    content = req.content
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="知识库不存在")
    kb = knowledge_bases[kb_id]
    kb["docs"].append(content)
    # 同步生成向量（懒加载 BGE 模型）
    vec = await asyncio.to_thread(_encode, [content])
    if vec:
        kb.setdefault("vectors", []).append(vec[0])
    return {"doc_index": len(kb["docs"]) - 1, "kb_id": kb_id}


@app.delete("/knowledge-base/{kb_id}/document/{doc_index}")
async def delete_document(kb_id: str, doc_index: int):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="知识库不存在")
    kb = knowledge_bases[kb_id]
    docs = kb["docs"]
    if doc_index < 0 or doc_index >= len(docs):
        raise HTTPException(status_code=404, detail="文档不存在")
    docs.pop(doc_index)
    vectors = kb.get("vectors", [])
    if doc_index < len(vectors):
        vectors.pop(doc_index)
    return {"message": "文档已删除"}


@app.delete("/knowledge-base/{kb_id}")
async def delete_kb(kb_id: str):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="知识库不存在")
    del knowledge_bases[kb_id]
    return {"message": f"知识库 {kb_id} 已删除"}


# ==================== 工作流 API ====================

class WorkflowNode(BaseModel):
    id: str
    type: str  # "start" | "llm" | "tool" | "condition" | "end" | "knowledge"
    label: str
    x: float
    y: float
    config: dict = {}

class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""

class WorkflowData(BaseModel):
    name: str
    description: str = ""
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []

class WorkflowRunRequest(_WithCreds):
    workflow_id: str
    input: str
    variables: dict = {}
    session_id: str = "default"

# 内存存储工作流
workflows: dict[str, dict] = {}

@app.post("/workflow")
async def create_workflow(data: WorkflowData):
    wf_id = str(uuid.uuid4())[:8]
    workflows[wf_id] = {
        "id": wf_id,
        "name": data.name,
        "description": data.description,
        "nodes": [n.model_dump() for n in data.nodes],
        "edges": [e.model_dump() for e in data.edges],
        "created_at": datetime.datetime.now().isoformat(),
    }
    return {"workflow_id": wf_id, "name": data.name}

@app.get("/workflow")
async def list_workflows():
    return {"workflows": list(workflows.values())}

@app.get("/workflow/{wf_id}")
async def get_workflow(wf_id: str):
    if wf_id not in workflows:
        raise HTTPException(status_code=404, detail="工作流不存在")
    return workflows[wf_id]

@app.put("/workflow/{wf_id}")
async def update_workflow(wf_id: str, data: WorkflowData):
    if wf_id not in workflows:
        raise HTTPException(status_code=404, detail="工作流不存在")
    workflows[wf_id].update({
        "name": data.name,
        "description": data.description,
        "nodes": [n.model_dump() for n in data.nodes],
        "edges": [e.model_dump() for e in data.edges],
    })
    return {"workflow_id": wf_id}

@app.delete("/workflow/{wf_id}")
async def delete_workflow(wf_id: str):
    if wf_id not in workflows:
        raise HTTPException(status_code=404, detail="工作流不存在")
    del workflows[wf_id]
    return {"message": f"工作流 {wf_id} 已删除"}

@app.post("/workflow/run")
async def run_workflow(req: WorkflowRunRequest):
    if req.workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="工作流不存在")
    wf = workflows[req.workflow_id]
    nodes = wf["nodes"]
    edges = wf["edges"]

    # 构建执行顺序（拓扑排序）
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["source"], []).append(e["target"])

    # 找 start 节点
    start_nodes = [n for n in nodes if n["type"] == "start"]
    if not start_nodes:
        raise HTTPException(status_code=400, detail="工作流缺少开始节点")

    # 简单线性执行：按拓扑顺序执行 LLM/Tool/Knowledge 节点
    visited = set()
    queue = [start_nodes[0]["id"]]
    execution_log = []
    current_input = req.input
    final_response = req.input

    # 构建变量替换表：{{变量名}} → 值，同时支持 {{input}}
    var_map = {k: str(v) for k, v in req.variables.items()}
    var_map["input"] = req.input

    def replace_vars(text: str) -> str:
        for k, v in var_map.items():
            text = text.replace("{{" + k + "}}", v)
        return text

    history = session_histories.get(req.session_id, [])
    system = SYSTEM_PROMPT

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node = next((n for n in nodes if n["id"] == node_id), None)
        if not node:
            continue

        ntype = node["type"]
        cfg = node.get("config", {})

        if ntype == "start":
            execution_log.append({"node": node["label"], "type": "start", "output": current_input})

        elif ntype == "llm":
            prompt_prefix = replace_vars(cfg.get("system_prompt", ""))
            var_map["input"] = current_input  # 更新 {{input}} 为当前输入
            msgs = ([{"role": "system", "content": (prompt_prefix + "\n" + system).strip()}]
                    + history
                    + [{"role": "user", "content": current_input}])
            resp_text, steps = await asyncio.to_thread(run_agent, req.api_key, req.base_url, cfg.get("model", "claude-sonnet-4-6"), msgs)
            current_input = resp_text
            final_response = resp_text
            execution_log.append({"node": node["label"], "type": "llm", "output": resp_text, "steps": steps})

        elif ntype == "tool":
            tool_name = cfg.get("tool_name", "")
            if tool_name in TOOL_FUNCTIONS:
                tool_args = dict(cfg.get("args", {}))
                var_map["input"] = current_input
                for k, v in tool_args.items():
                    if isinstance(v, str):
                        tool_args[k] = replace_vars(v)
                # LangChain Tool: 用 .invoke(args_dict) 而非 (**kwargs)
                result = TOOL_FUNCTIONS[tool_name].invoke(tool_args)
                current_input = result
                execution_log.append({"node": node["label"], "type": "tool", "output": result})

        elif ntype == "knowledge":
            kb_id = cfg.get("kb_id", "")
            if kb_id and kb_id in knowledge_bases:
                # LangChain Tool: 用 .invoke 派发
                result = search_knowledge_base.invoke({"kb_id": kb_id, "query": current_input})
                current_input = f"知识库检索结果：\n{result}\n\n原始问题：{req.input}"
                execution_log.append({"node": node["label"], "type": "knowledge", "output": result})

        elif ntype == "condition":
            expr = cfg.get("condition", "")
            result = False
            if ":" in expr:
                op, val = expr.split(":", 1)
                op = op.strip().lower()
                val = val.strip()
                text = current_input.lower()
                if op == "contains":
                    result = val.lower() in text
                elif op == "equals":
                    result = current_input.strip() == val
                elif op == "startswith":
                    result = text.startswith(val.lower())
            branch = "true" if result else "false"
            execution_log.append({"node": node["label"], "type": "condition", "output": f"条件: {expr} → {branch}"})
            # 只走匹配分支的边
            for e in edges:
                if e["source"] == node_id and e.get("label", "").lower() == branch and e["target"] not in visited:
                    queue.append(e["target"])
            continue

        elif ntype == "end":
            execution_log.append({"node": node["label"], "type": "end", "output": final_response})

        # 加入下一批节点
        for next_id in adj.get(node_id, []):
            if next_id not in visited:
                queue.append(next_id)

    # 更新会话历史
    history.append({"role": "user", "content": req.input})
    history.append({"role": "assistant", "content": final_response})
    session_histories[req.session_id] = history[-20:]

    return {
        "response": final_response,
        "execution_log": execution_log,
        "session_id": req.session_id,
    }


# ==================== Agent API ====================

class AgentData(BaseModel):
    name: str
    description: str = ""
    agent_type: str = "multi_agent"  # "multi_agent" | "autonomous"
    system_prompt: str = ""
    model_name: str = "claude-sonnet-4-6"
    greeting: str = ""
    suggested_questions: list[str] = []
    context_rounds: int = 2
    # 多智能体协同
    sub_agents: list[str] = []
    collaboration_mode: str = "sequential"  # "sequential" | "parallel" | "debate"
    # 自主规划
    available_tools: list[str] = []
    max_steps: int = 10

class AgentChatRequest(_WithCreds):
    message: str
    session_id: str = "default"

agents: dict[str, dict] = {}

@app.post("/agent")
async def create_agent_endpoint(data: AgentData):
    if data.agent_type == "multi_agent":
        for sid in data.sub_agents:
            if sid not in agents:
                raise HTTPException(status_code=400, detail=f"子 Agent {sid} 不存在")
    agent_id = str(uuid.uuid4())[:8]
    agents[agent_id] = {
        "id": agent_id,
        "name": data.name,
        "description": data.description,
        "agent_type": data.agent_type,
        "system_prompt": data.system_prompt,
        "model_name": data.model_name,
        "greeting": data.greeting,
        "suggested_questions": data.suggested_questions,
        "context_rounds": data.context_rounds,
        "sub_agents": data.sub_agents,
        "collaboration_mode": data.collaboration_mode,
        "available_tools": data.available_tools,
        "max_steps": data.max_steps,
        "created_at": datetime.datetime.now().isoformat(),
    }
    return {"agent_id": agent_id, "name": data.name}

@app.get("/agent")
async def list_agents():
    return {"agents": list(agents.values())}

@app.get("/agent/{agent_id}")
async def get_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    return agents[agent_id]

@app.put("/agent/{agent_id}")
async def update_agent(agent_id: str, data: AgentData):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    agents[agent_id].update({
        "name": data.name,
        "description": data.description,
        "agent_type": data.agent_type,
        "system_prompt": data.system_prompt,
        "model_name": data.model_name,
        "greeting": data.greeting,
        "suggested_questions": data.suggested_questions,
        "context_rounds": data.context_rounds,
        "sub_agents": data.sub_agents,
        "collaboration_mode": data.collaboration_mode,
        "available_tools": data.available_tools,
        "max_steps": data.max_steps,
    })
    return {"agent_id": agent_id}

@app.delete("/agent/{agent_id}")
async def delete_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    del agents[agent_id]
    return {"message": f"Agent {agent_id} 已删除"}


# ── 多智能体协同执行引擎 ──

def _call_sub_agent(sub_ag: dict, user_input: str, session_id: str, api_key: str, base_url: str) -> tuple[str, list]:
    """调用单个子 agent,返回 (回复文本, 工具步骤)。

    每个子 agent 用 deepagents.create_deep_agent 构造,自带 write_todos / 文件系统 / task 等内置原语,
    让子代理在 sequential/parallel/debate 流程内部也能自主规划而不是只跑 ReAct。
    """
    system = sub_ag.get("system_prompt") or SYSTEM_PROMPT
    history = session_histories.get(session_id, [])
    # system_prompt 走 create_deep_agent(system_prompt=...) 注入,messages 不再 prepend system 避免重复。
    messages = list(history) + [{"role": "user", "content": user_input}]
    model_name = sub_ag.get("model_name", "claude-sonnet-4-6")

    trace_cfg = _langsmith_config(
        run_name=f"sub_deep_agent:{sub_ag.get('name','sub')}",
        session_id=session_id,
        tags=["mode:multi_agent_sub", f"sub:{sub_ag.get('name','')}", f"model:{model_name}"],
        extra_metadata={"sub_agent_id": sub_ag.get("id", "")},
    )

    logger.info("[multi_agent_sub] 启动 deep sub-agent name=%s model=%s", sub_ag.get("name"), model_name)
    try:
        agent = _build_deep_agent(api_key, base_url, model_name, list(TOOLS), system)
        result = agent.invoke({"messages": messages}, config=trace_cfg)
    except Exception as e:
        logger.exception("[multi_agent_sub] deep agent invoke 异常 name=%s: %s", sub_ag.get("name"), e)
        return f"子代理执行失败: {type(e).__name__}: {e}", []

    summary = _summarize_deep_run(result)
    return summary["final"], summary["tool_steps"]


async def run_multi_agent_chat(ag: dict, user_input: str, session_id: str, api_key: str, base_url: str) -> dict:
    sub_ids = ag.get("sub_agents", [])
    mode = ag.get("collaboration_mode", "sequential")
    valid_subs = [agents[sid] for sid in sub_ids if sid in agents]

    # 没有子 agent 时，用主 agent 自身直接对话
    if not valid_subs:
        history = session_histories.get(session_id, [])
        system = ag.get("system_prompt") or SYSTEM_PROMPT
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]
        response_text, steps = await asyncio.to_thread(run_agent, api_key, base_url, ag.get("model_name", "claude-sonnet-4-6"), messages)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
        session_histories[session_id] = history[-20:]
        return {"response": response_text, "steps": steps, "multi_agent_log": [], "session_id": session_id}

    log = []

    if mode == "sequential":
        current_input = user_input
        for sub in valid_subs:
            text, steps = await asyncio.to_thread(_call_sub_agent, sub, current_input, session_id, api_key, base_url)
            log.append({"agent_name": sub["name"], "agent_id": sub["id"], "input": current_input, "output": text, "steps": steps})
            current_input = text
        final = current_input

    elif mode == "parallel":
        import concurrent.futures
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, _call_sub_agent, sub, user_input, session_id, api_key, base_url) for sub in valid_subs]
        results = await asyncio.gather(*tasks)
        for sub, (text, steps) in zip(valid_subs, results):
            log.append({"agent_name": sub["name"], "agent_id": sub["id"], "input": user_input, "output": text, "steps": steps})
        # 用主 agent 汇总
        summary_prompt = f"以下是多个 AI 助手对用户问题的回答，请综合所有回答给出最终答案。\n\n用户问题: {user_input}\n\n"
        for entry in log:
            summary_prompt += f"【{entry['agent_name']}】的回答:\n{entry['output']}\n\n"
        system = ag.get("system_prompt") or "你是一个汇总助手，请综合多个助手的回答给出最终答案。"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": summary_prompt}]
        final, _ = await asyncio.to_thread(run_agent, api_key, base_url, ag["model_name"], msgs)

    elif mode == "debate":
        # 第一轮：各自回答
        round1 = {}
        for sub in valid_subs:
            text, steps = await asyncio.to_thread(_call_sub_agent, sub, user_input, session_id, api_key, base_url)
            round1[sub["id"]] = text
            log.append({"agent_name": sub["name"], "agent_id": sub["id"], "round": 1, "output": text, "steps": steps})
        # 第二轮：看到其他人的观点后再回答
        for sub in valid_subs:
            others = "\n\n".join(f"【{agents[sid]['name']}】的观点: {round1[sid]}" for sid in round1 if sid != sub["id"])
            debate_input = f"用户问题: {user_input}\n\n其他助手的观点:\n{others}\n\n请在看到其他观点后，给出你的最终回答。你可以坚持自己的观点，也可以修正。"
            text, steps = await asyncio.to_thread(_call_sub_agent, sub, debate_input, session_id, api_key, base_url)
            log.append({"agent_name": sub["name"], "agent_id": sub["id"], "round": 2, "output": text, "steps": steps})
        # 主 agent 裁决
        all_views = "\n\n".join(f"【{e['agent_name']}】(第{e['round']}轮): {e['output']}" for e in log)
        judge_prompt = f"以下是多个助手经过两轮辩论后的观点。请作为裁判，综合所有观点给出最终结论。\n\n用户问题: {user_input}\n\n{all_views}"
        system = ag.get("system_prompt") or "你是辩论裁判，请综合各方观点给出公正的最终结论。"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": judge_prompt}]
        final, _ = await asyncio.to_thread(run_agent, api_key, base_url, ag["model_name"], msgs)
    else:
        final = "不支持的协作模式"

    history = session_histories.get(session_id, [])
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final})
    session_histories[session_id] = history[-20:]
    return {"response": final, "steps": [], "multi_agent_log": log, "session_id": session_id}


# ── 自主规划执行引擎 (基于 deepagents.create_deep_agent) ──


def _build_deep_agent(api_key: str, base_url: str, model_name: str, tools: list, system_prompt: str):
    """构造 deep agent。

    deep agent 自带 write_todos / 文件系统 / task / execute 等内置工具,实现"规划 + 子代理"原语,
    替代旧版手写的 PLAN/REFLECT/replan 循环。
    """
    model = _build_chat_model(api_key, base_url, model_name)
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


def _summarize_deep_run(result: dict) -> dict:
    """从 deep agent invoke 结果里抽取 todos / files / final_text 摘要,供前端日志展示。"""
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    final = ""
    if msgs:
        final = getattr(msgs[-1], "content", "") or ""
    files = result.get("files", {}) if isinstance(result, dict) else {}
    todos = result.get("todos", []) if isinstance(result, dict) else []
    return {
        "final": final,
        "todos": todos,
        "file_keys": list(files.keys()) if isinstance(files, dict) else [],
        "tool_steps": _extract_steps_from_messages(msgs),
    }


async def run_autonomous_chat(ag: dict, user_input: str, session_id: str, api_key: str, base_url: str) -> dict:
    """autonomous 模式: 用 deepagents.create_deep_agent 替代旧版手写 PLAN/REFLECT/replan 循环。

    保留语义:
    - available_tools 白名单仍然生效(过滤用户 TOOLS,不影响 deep agent 内置工具)
    - max_steps 转换为 recursion_limit(deep agent 一个用户步骤可能涉及多次 tool call)
    - 返回字段保持 {response, steps, autonomous_log, session_id} 形态
    """
    available = ag.get("available_tools", [])
    max_steps = ag.get("max_steps", 10)
    allowed_tools = [t for t in TOOLS if t.name in available] if available else list(TOOLS)
    system = ag.get("system_prompt") or SYSTEM_PROMPT

    history = session_histories.get(session_id, [])
    # 注意: system_prompt 已经通过 _build_deep_agent → create_deep_agent(system_prompt=...) 注入,
    # messages 里不要再 prepend system,否则会和 SDK 默认 prompt 一起重复传给模型。
    messages = list(history) + [{"role": "user", "content": user_input}]

    trace_cfg = _langsmith_config(
        run_name=f"deep_agent:{ag.get('name','autonomous')}",
        session_id=session_id,
        tags=["endpoint:agent_chat", "mode:autonomous", f"model:{ag.get('model_name','')}"],
        extra_metadata={"agent_id": ag.get("id", ""), "max_steps": max_steps},
        recursion_limit=max(25, max_steps * 6),
    )

    logger.info("[autonomous] 启动 deep agent agent_id=%s max_steps=%d tools=%d",
                ag.get("id"), max_steps, len(allowed_tools))

    def _run():
        agent = _build_deep_agent(api_key, base_url, ag["model_name"], allowed_tools, system)
        return agent.invoke({"messages": messages}, config=trace_cfg)

    try:
        result = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("[autonomous] deep agent invoke 异常: %s", e)
        err = f"自主执行失败: {type(e).__name__}: {e}"
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": err})
        session_histories[session_id] = history[-20:]
        return {"response": err, "steps": [], "autonomous_log": [{"phase": "error", "detail": str(e)}], "session_id": session_id}

    summary = _summarize_deep_run(result)
    final = summary["final"]
    logger.info("[autonomous] 完成 final_len=%d steps=%d todos=%d files=%d",
                len(final), len(summary["tool_steps"]), len(summary["todos"]), len(summary["file_keys"]))

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final})
    session_histories[session_id] = history[-20:]

    return {
        "response": final,
        "steps": summary["tool_steps"],
        "autonomous_log": [{
            "phase": "deep_agent_run",
            "todos": summary["todos"],
            "file_keys": summary["file_keys"],
            "tool_steps": summary["tool_steps"],
        }],
        "session_id": session_id,
    }


@app.post("/agent/{agent_id}/chat")
async def agent_chat(agent_id: str, req: AgentChatRequest):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    ag = agents[agent_id]
    atype = ag.get("agent_type", "multi_agent")

    if atype == "multi_agent":
        return await run_multi_agent_chat(ag, req.message, req.session_id, req.api_key, req.base_url)
    elif atype == "autonomous":
        return await run_autonomous_chat(ag, req.message, req.session_id, req.api_key, req.base_url)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的 Agent 类型: {atype}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)