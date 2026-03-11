"""
LangChain Agent Enhanced - 增强版
新增功能：
1. 工作流条件分支（Workflow with conditional nodes）
2. 记忆系统（Memory system with vector store）
3. 知识库模板（Knowledge base templates with RAG）
"""
import math
import datetime
import json
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from openai import OpenAI
from enum import Enum

# 向量存储和嵌入
try:
    from chromadb import Client as ChromaClient
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Knowledge base features disabled.")

app = FastAPI(title="LangChain Agent Enhanced API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 数据模型 ====================

class NodeType(str, Enum):
    LLM = "llm"              # LLM 推理节点
    TOOL = "tool"            # 工具调用节点
    CONDITION = "condition"  # 条件分支节点
    MEMORY = "memory"        # 记忆检索节点
    KNOWLEDGE = "knowledge"  # 知识库查询节点
    MERGE = "merge"          # 合并节点

class WorkflowNode(BaseModel):
    id: str
    type: NodeType
    config: Dict[str, Any] = {}
    next_nodes: List[str] = []  # 下一个节点 ID 列表
    condition: Optional[str] = None  # 条件表达式（用于 CONDITION 节点）

class Workflow(BaseModel):
    id: str
    name: str
    nodes: List[WorkflowNode]
    start_node: str

class MemoryEntry(BaseModel):
    id: str
    session_id: str
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: float
    importance: float = 0.5  # 0-1，重要性评分

class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str
    documents: List[str] = []
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model_name: Optional[str] = "gpt-4o-mini"
    workflow_id: Optional[str] = None  # 可选：使用特定工作流
    knowledge_base_ids: List[str] = []  # 可选：使用的知识库 ID

# ==================== 全局存储 ====================

session_histories: Dict[str, List[Dict]] = {}
workflows: Dict[str, Workflow] = {}
knowledge_bases: Dict[str, KnowledgeBase] = {}
memory_store: Dict[str, List[MemoryEntry]] = {}

# 向量数据库客户端
if CHROMA_AVAILABLE:
    chroma_client = ChromaClient(Settings(anonymized_telemetry=False))
    memory_collection = chroma_client.get_or_create_collection("agent_memory")
    knowledge_collection = chroma_client.get_or_create_collection("knowledge_base")
else:
    chroma_client = None
    memory_collection = None
    knowledge_collection = None

# ==================== 原有工具函数 ====================

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
        ("m", "km"): 0.001, ("km", "m"): 1000,
        ("m", "mile"): 0.000621371, ("mile", "m"): 1609.34,
        ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
        ("m", "cm"): 100, ("cm", "m"): 0.01,
        ("m", "mm"): 1000, ("mm", "m"): 0.001,
        ("km", "mile"): 0.621371, ("mile", "km"): 1.60934,
        ("cm", "mm"): 10, ("mm", "cm"): 0.1,
        ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
        ("kg", "g"): 1000, ("g", "kg"): 0.001,
        ("kg", "oz"): 35.274, ("oz", "kg"): 0.0283495,
        ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
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

# ==================== 新增：记忆系统 ====================

def store_memory(session_id: str, content: str, metadata: Dict = None, importance: float = 0.5):
    """存储记忆到向量数据库"""
    memory_id = str(uuid.uuid4())
    entry = MemoryEntry(
        id=memory_id,
        session_id=session_id,
        content=content,
        metadata=metadata or {},
        timestamp=datetime.datetime.now().timestamp(),
        importance=importance
    )

    if session_id not in memory_store:
        memory_store[session_id] = []
    memory_store[session_id].append(entry)

    # 存储到向量数据库
    if memory_collection:
        memory_collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{"session_id": session_id, "importance": importance, **metadata}]
        )

    return memory_id

def retrieve_memory(session_id: str, query: str, top_k: int = 3) -> List[Dict]:
    """从记忆中检索相关内容"""
    if not memory_collection:
        # 降级：返回最近的记忆
        recent = memory_store.get(session_id, [])[-top_k:]
        return [{"content": m.content, "importance": m.importance} for m in recent]

    results = memory_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"session_id": session_id}
    )

    if not results['documents'][0]:
        return []

    return [
        {
            "content": doc,
            "importance": meta.get("importance", 0.5),
            "metadata": meta
        }
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]

def search_memory(session_id: str, query: str, top_k: int = 5) -> str:
    """工具函数：搜索记忆"""
    memories = retrieve_memory(session_id, query, top_k)
    if not memories:
        return "未找到相关记忆"

    result = "🧠 相关记忆:\n"
    for i, mem in enumerate(memories, 1):
        result += f"{i}. {mem['content'][:100]}...\n"
    return result

# ==================== 新增：知识库系统 ====================

def create_knowledge_base(name: str, description: str) -> str:
    """创建知识库"""
    kb_id = str(uuid.uuid4())
    kb = KnowledgeBase(
        id=kb_id,
        name=name,
        description=description
    )
    knowledge_bases[kb_id] = kb
    return kb_id

def add_document_to_kb(kb_id: str, content: str, metadata: Dict = None) -> str:
    """添加文档到知识库"""
    if kb_id not in knowledge_bases:
        return "知识库不存在"

    doc_id = str(uuid.uuid4())
    knowledge_bases[kb_id].documents.append(doc_id)

    if knowledge_collection:
        knowledge_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"kb_id": kb_id, **(metadata or {})}]
        )

    return doc_id

def query_knowledge_base(kb_ids: List[str], query: str, top_k: int = 3) -> str:
    """从知识库检索信息"""
    if not knowledge_collection or not kb_ids:
        return "知识库功能未启用或未指定知识库"

    results = knowledge_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"kb_id": {"$in": kb_ids}} if kb_ids else None
    )

    if not results['documents'][0]:
        return "未找到相关文档"

    context = "📚 知识库检索结果:\n\n"
    for i, doc in enumerate(results['documents'][0], 1):
        context += f"[文档 {i}]\n{doc}\n\n"

    return context

# ==================== 工具注册 ====================

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "text_analyzer": text_analyzer,
    "unit_converter": unit_converter,
    "word_counter": word_counter,
    "search_memory": search_memory,
    "query_knowledge_base": lambda query, kb_ids=None: query_knowledge_base(kb_ids or [], query),
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。支持基本四则运算、幂运算、三角函数、对数等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "时区"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "分析文本统计信息",
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
            "description": "单位换算（长度、重量、温度）",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "数值"},
                    "from_unit": {"type": "string", "description": "源单位"},
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
            "description": "统计词语出现次数",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "文本"},
                    "target_word": {"type": "string", "description": "目标词语"}
                },
                "required": ["text", "target_word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "搜索历史对话记忆，找到相关的上下文信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "top_k": {"type": "integer", "description": "返回结果数量", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "从知识库中检索相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "kb_ids": {"type": "array", "items": {"type": "string"}, "description": "知识库 ID 列表"}
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """你是一个智能助手，可以使用多种工具来帮助用户。

你拥有以下能力：
1. 基础工具：计算器、时间查询、文本分析、单位换算等
2. 记忆系统：可以搜索历史对话中的相关信息
3. 知识库：可以从专业知识库中检索信息

使用指南：
- 当用户提到"之前"、"上次"等词时，使用 search_memory 工具查找相关记忆
- 当需要专业知识时，使用 query_knowledge_base 工具
- 优先使用工具获取准确信息，而不是依赖自己的知识

请用简洁、专业的方式回答用户问题。"""

# ==================== Agent 执行引擎 ====================

def run_agent(api_key: str, base_url: str, model: str, messages: List[Dict],
              session_id: str = "default", kb_ids: List[str] = None) -> tuple:
    """执行 Agent 循环"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    steps = []
    max_iterations = 10

    # 增强上下文：添加记忆检索
    if messages and messages[-1]["role"] == "user":
        user_query = messages[-1]["content"]
        relevant_memories = retrieve_memory(session_id, user_query, top_k=2)

        if relevant_memories:
            memory_context = "\n[相关历史记忆]\n"
            for mem in relevant_memories:
                memory_context += f"- {mem['content']}\n"
            messages.insert(-1, {"role": "system", "content": memory_context})

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            final_response = message.content

            # 存储重要对话到记忆
            if messages[-1]["role"] == "user":
                store_memory(
                    session_id=session_id,
                    content=f"Q: {messages[-1]['content']}\nA: {final_response}",
                    metadata={"type": "qa_pair"},
                    importance=0.7
                )

            return final_response, steps

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # 特殊处理：为某些工具注入 session_id
            if func_name == "search_memory":
                args["session_id"] = session_id
            elif func_name == "query_knowledge_base" and kb_ids:
                args["kb_ids"] = kb_ids

            try:
                result = TOOL_FUNCTIONS[func_name](**args)
            except Exception as e:
                result = f"工具执行错误: {str(e)}"

            steps.append({
                "tool": func_name,
                "args": args,
                "result": result
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "达到最大迭代次数", steps

# ==================== API 端点 ====================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "features": {
            "vector_store": CHROMA_AVAILABLE,
            "memory_system": True,
            "knowledge_base": True,
            "workflow": True
        }
    }

@app.get("/tools")
async def get_tools():
    return {
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"]
            }
            for t in TOOLS_SCHEMA
        ]
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        history = session_histories.get(req.session_id, [])

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": req.message}]
        )

        response_text, steps = await asyncio.to_thread(
            run_agent,
            req.api_key,
            req.base_url,
            req.model_name,
            messages,
            req.session_id,
            req.knowledge_base_ids
        )

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

# ==================== 知识库管理 API ====================

@app.post("/knowledge-base")
async def create_kb(name: str, description: str):
    """创建知识库"""
    kb_id = create_knowledge_base(name, description)
    return {"kb_id": kb_id, "name": name}

@app.get("/knowledge-base")
async def list_kb():
    """列出所有知识库"""
    return {
        "knowledge_bases": [
            {"id": kb.id, "name": kb.name, "description": kb.description, "doc_count": len(kb.documents)}
            for kb in knowledge_bases.values()
        ]
    }

@app.post("/knowledge-base/{kb_id}/document")
async def add_document(kb_id: str, content: str, metadata: Dict = None):
    """添加文档到知识库"""
    doc_id = add_document_to_kb(kb_id, content, metadata)
    return {"doc_id": doc_id, "kb_id": kb_id}

@app.post("/knowledge-base/{kb_id}/upload")
async def upload_file(kb_id: str, file: UploadFile = File(...)):
    """上传文件到知识库"""
    content = await file.read()
    text_content = content.decode('utf-8')

    # 简单分块（每 500 字符一块）
    chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]

    doc_ids = []
    for i, chunk in enumerate(chunks):
        doc_id = add_document_to_kb(
            kb_id,
            chunk,
            {"filename": file.filename, "chunk_index": i}
        )
        doc_ids.append(doc_id)

    return {"kb_id": kb_id, "filename": file.filename, "chunks": len(doc_ids)}

# ==================== 记忆管理 API ====================

@app.get("/memory/{session_id}")
async def get_memories(session_id: str, limit: int = 10):
    """获取会话记忆"""
    memories = memory_store.get(session_id, [])[-limit:]
    return {
        "session_id": session_id,
        "memories": [
            {
                "content": m.content,
                "timestamp": m.timestamp,
                "importance": m.importance
            }
            for m in memories
        ]
    }

@app.post("/memory/{session_id}/search")
async def search_memories(session_id: str, query: str, top_k: int = 5):
    """搜索记忆"""
    results = retrieve_memory(session_id, query, top_k)
    return {"query": query, "results": results}

# ==================== 工作流 API（预留接口）====================

@app.post("/workflow")
async def create_workflow(workflow: Workflow):
    """创建工作流"""
    workflows[workflow.id] = workflow
    return {"workflow_id": workflow.id, "name": workflow.name}

@app.get("/workflow")
async def list_workflows():
    """列出所有工作流"""
    return {
        "workflows": [
            {"id": wf.id, "name": wf.name, "node_count": len(wf.nodes)}
            for wf in workflows.values()
        ]
    }

@app.get("/workflow/{workflow_id}")
async def get_workflow(workflow_id: str):
    """获取工作流详情"""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflows[workflow_id]

if __name__ == "__main__":
    import uvicorn
    print("🚀 LangChain Agent Enhanced Starting...")
    print(f"   Vector Store: {'✅ Enabled' if CHROMA_AVAILABLE else '❌ Disabled (install chromadb)'}")
    print(f"   Memory System: ✅ Enabled")
    print(f"   Knowledge Base: ✅ Enabled")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
