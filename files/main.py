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


# 工具分发表
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "text_analyzer": text_analyzer,
    "unit_converter": unit_converter,
    "word_counter": word_counter,
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
    }
]

SYSTEM_PROMPT = """你是一个智能助手，具备多种实用工具能力。你可以帮助用户进行：
- 数学计算（使用 calculator 工具）
- 查询当前时间（使用 get_current_time 工具）
- 文本分析（使用 text_analyzer 工具）
- 单位换算（使用 unit_converter 工具）
- 词语统计（使用 word_counter 工具）

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

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    api_key: str
    base_url: str = "https://api.penguinsaichat.dpdns.org/v1"
    model_name: str = "claude-sonnet-4-6"


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

        # 构造完整消息列表（SystemMessage + 历史 + 当前问题）
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
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
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    if session_id in session_histories:
        del session_histories[session_id]
    return {"message": f"Session {session_id} cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)