"""
LangChain Agent Backend - FastAPI
手动工具调用循环实现，兼容第三方/中转 OpenAI 格式接口
不依赖 LangGraph，避免 model_dump / tool_calls 格式兼容问题
"""
# 2024-06-01 by ChatGPT
import os
import math
import datetime
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import urllib.request
import urllib.parse
import ssl

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE
from openai import OpenAI

app = FastAPI(title="LangChain Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Tools 实现 ====================

def calculator(expression: str) -> str:
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round, "pow": pow})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
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

SYSTEM_PROMPT = """你是一个智能助手，具备多种实用工具能力。你可以帮助用户进行：
- 数学计算（使用 calculator 工具）
- 查询当前时间（使用 get_current_time 工具）
- 文本分析（使用 text_analyzer 工具）
- 单位换算（使用 unit_converter 工具）
- 词语统计（使用 word_counter 工具）
- 查询实时天气（使用 get_weather 工具）
- 搜索知识库（使用 search_knowledge_base 工具，需提供 kb_id）

请根据用户的问题，灵活使用工具，给出准确、有用的回答。
回答时请用中文，并保持友好、专业的语气。
如果不需要工具，直接回答即可。"""

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

# 硬编码配置，不在前端暴露
API_KEY = "REDACTED_API_KEY"
BASE_URL = "https://api.penguinsaichat.dpdns.org/v1"


class ChatRequest(BaseModel):
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
            run_agent, API_KEY, BASE_URL, req.model_name, messages
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
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)