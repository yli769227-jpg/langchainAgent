"""
LangChain Agent Backend - FastAPI
手动工具调用循环实现，兼容第三方/中转 OpenAI 格式接口
不依赖 LangGraph，避免 model_dump / tool_calls 格式兼容问题
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("agent")


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

from openai import OpenAI

app = FastAPI(title="LangChain Agent API", version="1.0.0")

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


def calculator(expression: str) -> str:
    if len(expression) > 500:
        return "计算错误: 表达式过长（上限 500 字符）"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _calc_eval(tree)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    now = datetime.datetime.now()
    return (
        f"当前时间（北京时间）: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期: {['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}\n"
        f"今年第 {now.timetuple().tm_yday} 天"
    )


def text_analyzer(text: str) -> str:
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


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
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


def word_counter(text: str, target_word: str) -> str:
    count = text.lower().count(target_word.lower())
    return f"词语 '{target_word}' 在文本中出现了 {count} 次"


def get_weather(city: str) -> str:
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


def search_knowledge_base(kb_id: str, query: str, top_k: int = 3) -> str:
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


# 工具分发表
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "text_analyzer": text_analyzer,
    "unit_converter": unit_converter,
    "word_counter": word_counter,
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base,
}

# ==================== OpenAI tools 格式定义 ====================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。支持基本四则运算、幂运算、三角函数、对数等。示例: '2 + 3 * 4', 'sqrt(16)'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式字符串"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间（北京时间）",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "时区，默认 Asia/Shanghai"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "分析文本统计信息：字符数、词数、行数、中文字符数等",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要分析的文本"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "单位换算。支持长度(m,km,mile,ft,cm,mm)、重量(kg,lb,g,oz)、温度(celsius,fahrenheit,kelvin)",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "要换算的数值"},
                    "from_unit": {"type": "string", "description": "原单位"},
                    "to_unit": {"type": "string", "description": "目标单位"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_counter",
            "description": "在文本中统计特定词语出现的次数（不区分大小写）",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要搜索的文本"},
                    "target_word": {"type": "string", "description": "要统计的词语"}
                },
                "required": ["text", "target_word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气，包括温度、湿度、风速、天气状况等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，支持中文或英文，如 '北京'、'Shanghai'、'London'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在指定知识库中搜索相关内容，用于回答基于知识库的问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_id": {"type": "string", "description": "知识库 ID"},
                    "query": {"type": "string", "description": "搜索查询内容"},
                    "top_k": {"type": "integer", "description": "返回最相关的文档数量，默认 3", "default": 3}
                },
                "required": ["kb_id", "query"]
            }
        }
    }
]

SYSTEM_PROMPT = """你是一个智能助手。请根据用户的问题给出准确、有用的回答。
回答时请用中文，并保持友好、专业的语气。
如果需要使用工具来回答问题，请直接使用，不要向用户列举你的工具能力。"""

# 全局会话历史
session_histories: dict[str, list] = {}


def run_agent(api_key: str, base_url: str, model_name: str, messages: list) -> tuple[str, list]:
    """
    手动工具调用循环，完全基于原生 OpenAI SDK。
    兼容所有支持 OpenAI 格式的中转服务。
    返回 (最终回复文本, 工具调用步骤列表)
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    steps = []
    # 最多循环 10 次，防止死循环
    for _ in range(10):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # 没有工具调用，直接返回
        if not msg.tool_calls:
            return msg.content, steps

        # 有工具调用，逐个执行
        # 把 assistant 消息加入历史
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]
        })

        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except Exception:
                func_args = {}

            # 执行工具
            if func_name in TOOL_FUNCTIONS:
                tool_result = TOOL_FUNCTIONS[func_name](**func_args)
            else:
                tool_result = f"未知工具: {func_name}"

            steps.append({
                "tool": func_name,
                "input": func_args,
                "output": tool_result,
            })

            # 把工具结果加入历史
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # 超过最大循环次数，强制返回
    return "抱歉，处理过程中遇到了问题，请重试。", steps


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

        response_text, steps = await asyncio.to_thread(
            run_agent, req.api_key, req.base_url, req.model_name, messages
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
        client = OpenAI(api_key=req.api_key, base_url=req.base_url)
        steps = []
        for _ in range(10):
            stream = client.chat.completions.create(
                model=req.model_name,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                stream=True,
            )
            content_parts = []
            tool_calls_map = {}
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield f"data: {json.dumps({'type':'token','content':delta.content}, ensure_ascii=False)}\n\n"
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["arguments"] += tc.function.arguments

            full_content = "".join(content_parts)

            if not tool_calls_map:
                # No tool calls, done
                history.append({"role": "user", "content": req.message})
                history.append({"role": "assistant", "content": full_content})
                session_histories[req.session_id] = history[-20:]
                yield f"data: {json.dumps({'type':'done','steps':steps}, ensure_ascii=False)}\n\n"
                return

            # Process tool calls
            assistant_msg = {
                "role": "assistant",
                "content": full_content or "",
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_map.values()
                ]
            }
            messages.append(assistant_msg)

            for tc in tool_calls_map.values():
                func_name = tc["name"]
                try:
                    func_args = json.loads(tc["arguments"])
                except Exception:
                    func_args = {}
                if func_name in TOOL_FUNCTIONS:
                    tool_result = TOOL_FUNCTIONS[func_name](**func_args)
                else:
                    tool_result = f"未知工具: {func_name}"
                step = {"tool": func_name, "input": func_args, "output": tool_result}
                steps.append(step)
                yield f"data: {json.dumps({'type':'tool','step':step}, ensure_ascii=False)}\n\n"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

        yield f"data: {json.dumps({'type':'done','steps':steps}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
async def clear_history(session_id: str):
    if session_id in session_histories:
        del session_histories[session_id]
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
                result = TOOL_FUNCTIONS[tool_name](**tool_args)
                current_input = result
                execution_log.append({"node": node["label"], "type": "tool", "output": result})

        elif ntype == "knowledge":
            kb_id = cfg.get("kb_id", "")
            if kb_id and kb_id in knowledge_bases:
                result = search_knowledge_base(kb_id, current_input)
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
async def create_agent(data: AgentData):
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
    """调用单个子 agent，返回 (回复文本, 工具步骤)"""
    system = sub_ag.get("system_prompt") or SYSTEM_PROMPT
    history = session_histories.get(session_id, [])
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_input}]
    return run_agent(api_key, base_url, sub_ag.get("model_name", "claude-sonnet-4-6"), messages)


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


# ── 自主规划执行引擎 ──

PLAN_PROMPT = """你是一个任务规划专家。请将用户的任务拆解为具体的执行步骤。
你必须以 JSON 格式输出，格式如下：
{"steps": [{"step": 1, "description": "步骤描述", "tool": "工具名称或null"}]}

可用工具: {tools}

注意：
- 每个步骤应该是一个具体的、可执行的操作
- tool 字段如果不需要工具则填 null
- 步骤数量不要超过 {max_steps} 步
- 只输出 JSON，不要输出其他内容"""

REFLECT_PROMPT = """你是一个任务执行评估专家。请评估当前步骤的执行结果，判断是否需要调整后续计划。

原始任务: {task}
当前计划: {plan}
已完成步骤: {completed}
当前步骤结果: {current_result}

请以 JSON 格式回答：
{{"assessment": "对当前结果的评估", "need_replan": true/false, "reason": "原因"}}

只输出 JSON。"""


async def run_autonomous_chat(ag: dict, user_input: str, session_id: str, api_key: str, base_url: str) -> dict:
    available = ag.get("available_tools", [])
    max_steps = ag.get("max_steps", 10)
    tool_names = ", ".join(available) if available else "无"
    execution_log = []

    # 1. 规划阶段
    plan_system = PLAN_PROMPT.format(tools=tool_names, max_steps=max_steps)
    plan_msgs = [{"role": "system", "content": plan_system}, {"role": "user", "content": user_input}]
    plan_text, _ = await asyncio.to_thread(run_agent, api_key, base_url, ag["model_name"], plan_msgs)

    try:
        # 提取 JSON（兼容 markdown code block）
        clean = plan_text.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        plan = json.loads(clean)
    except Exception:
        plan = {"steps": [{"step": 1, "description": user_input, "tool": None}]}

    execution_log.append({"phase": "plan", "content": plan})
    steps_plan = plan.get("steps", [])
    completed_results = []

    # 2. 执行 + 反思循环
    step_count = 0
    i = 0
    while i < len(steps_plan) and step_count < max_steps:
        step_info = steps_plan[i]
        step_desc = step_info.get("description", "")
        step_tool = step_info.get("tool")

        # 构建执行 prompt
        context = "\n".join(f"步骤{j+1}结果: {r}" for j, r in enumerate(completed_results))
        exec_prompt = f"请执行以下任务步骤:\n{step_desc}"
        if context:
            exec_prompt = f"之前步骤的结果:\n{context}\n\n{exec_prompt}"

        system = ag.get("system_prompt") or SYSTEM_PROMPT
        # 如果步骤指定了工具且在可用列表中，构建带工具的消息
        tool_schemas = [t for t in TOOLS_SCHEMA if t["function"]["name"] in available] if available else TOOLS_SCHEMA
        exec_msgs = [{"role": "system", "content": system}, {"role": "user", "content": exec_prompt}]

        # 执行（带工具调用）
        client = OpenAI(api_key=api_key, base_url=base_url)
        exec_steps = []
        for _ in range(5):
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=ag["model_name"], messages=exec_msgs,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto" if tool_schemas else None,
                )
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                step_result = msg.content
                break
            exec_msgs.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
            })
            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                result = TOOL_FUNCTIONS[fn](**args) if fn in TOOL_FUNCTIONS else f"未知工具: {fn}"
                exec_steps.append({"tool": fn, "input": args, "output": result})
                exec_msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            step_result = "步骤执行超时"

        completed_results.append(step_result)
        execution_log.append({
            "phase": "execute", "step": step_info.get("step", i+1),
            "description": step_desc, "result": step_result, "tool_steps": exec_steps
        })

        # 3. 反思阶段
        reflect_system = REFLECT_PROMPT.format(
            task=user_input, plan=json.dumps(steps_plan, ensure_ascii=False),
            completed=json.dumps(completed_results, ensure_ascii=False), current_result=step_result
        )
        reflect_msgs = [{"role": "system", "content": reflect_system}, {"role": "user", "content": "请评估"}]
        reflect_text, _ = await asyncio.to_thread(run_agent, api_key, base_url, ag["model_name"], reflect_msgs)

        try:
            rclean = reflect_text.strip()
            if "```" in rclean:
                rclean = rclean.split("```")[1]
                if rclean.startswith("json"):
                    rclean = rclean[4:]
                rclean = rclean.strip()
            reflection = json.loads(rclean)
        except Exception:
            reflection = {"assessment": reflect_text, "need_replan": False, "reason": ""}

        execution_log.append({"phase": "reflect", "step": step_info.get("step", i+1), "content": reflection})

        # 4. 重规划
        if reflection.get("need_replan"):
            replan_prompt = f"原始任务: {user_input}\n已完成步骤结果: {json.dumps(completed_results, ensure_ascii=False)}\n反思: {reflection.get('reason', '')}\n\n请重新规划剩余步骤。"
            replan_msgs = [{"role": "system", "content": plan_system}, {"role": "user", "content": replan_prompt}]
            replan_text, _ = await asyncio.to_thread(run_agent, api_key, base_url, ag["model_name"], replan_msgs)
            try:
                rp_clean = replan_text.strip()
                if "```" in rp_clean:
                    rp_clean = rp_clean.split("```")[1]
                    if rp_clean.startswith("json"):
                        rp_clean = rp_clean[4:]
                    rp_clean = rp_clean.strip()
                new_plan = json.loads(rp_clean)
                steps_plan = new_plan.get("steps", [])
                i = 0
                execution_log.append({"phase": "replan", "content": new_plan})
            except Exception:
                i += 1
        else:
            i += 1
        step_count += 1

    # 最终汇总
    summary_prompt = f"原始任务: {user_input}\n\n执行结果:\n" + "\n".join(f"步骤{j+1}: {r}" for j, r in enumerate(completed_results)) + "\n\n请给出最终的综合回答。"
    system = ag.get("system_prompt") or SYSTEM_PROMPT
    summary_msgs = [{"role": "system", "content": system}, {"role": "user", "content": summary_prompt}]
    final, _ = await asyncio.to_thread(run_agent, api_key, base_url, ag["model_name"], summary_msgs)

    history = session_histories.get(session_id, [])
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final})
    session_histories[session_id] = history[-20:]
    return {"response": final, "steps": [], "autonomous_log": execution_log, "session_id": session_id}


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