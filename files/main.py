"""
LangChain Agent Backend - FastAPI

真接入 LangChain 1.2 + LangGraph 1.1 栈：
- 工具用 @tool 装饰，schema 自动派生(在 tools/ 子包里逐个文件维护)
- run_agent 走 langchain.agents.create_agent (LangGraph 状态机)
- /chat/stream 用 agent.astream_events 流式
- 兼容第三方/中转 OpenAI 格式接口（任意 base_url）
- 可观测: LangSmith trace + Langfuse callback,二者独立,各自由 env 驱动
"""

import os
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

# LangChain / LangGraph 栈
from langchain.agents import create_agent
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent, SubAgent

# 在任何 langchain/langsmith import 触发 env 读取前先加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 工具拆出到 tools/ 子包,共享运行时状态(知识库 / BGE / SSL) 放 runtime.py
# main.py 顶部 re-export 这些名字,保持原 main.* 访问路径兼容 tests/ 和外部调用方。
from runtime import (
    _bge_model,  # 兼容 import: 测试和旧调用方可能引用
    _cosine,
    _encode,
    _get_bge_model,
    _keyword_score,
    _ssl_ctx,
    knowledge_bases,
)
from tools import (
    ALL_TOOLS,
    calculator,
    fetch_url,
    get_current_time,
    get_weather,
    search_knowledge_base,
    text_analyzer,
    unit_converter,
    web_search,
    word_counter,
)
from tools.calculator import _calc_eval  # tests/test_calc_safety.py 用 main._calc_eval

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
    if _langsmith_enabled():
        cfg["run_name"] = run_name
        cfg["tags"] = list(tags or [])
        metadata: dict = {}
        if session_id:
            metadata["session_id"] = session_id
        if extra_metadata:
            metadata.update(extra_metadata)
        if metadata:
            cfg["metadata"] = metadata
    # Langfuse 独立通道:env 就绪就把 CallbackHandler 加到 callbacks,
    # 即使 LangSmith 没开也能单独用 Langfuse 看 trace。
    _inject_langfuse_callback(cfg)
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


# ==================== Langfuse 可选 callback ====================
# Langfuse 与 LangSmith 独立:LangSmith 走 langchain 内置 tracer(env LANGSMITH_*),
# Langfuse 走 langchain CallbackHandler,我们注入到 invoke config["callbacks"]。
# 任一缺 PUBLIC_KEY/SECRET_KEY 就跳过,不报错。装好 langfuse 包但没配 key 也跳过。


def _langfuse_enabled() -> bool:
    """同时配了 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY 才视作启用。"""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


_langfuse_handler = None
_langfuse_init_failed = False


def _get_langfuse_handler():
    """懒加载 Langfuse CallbackHandler。env 缺 key / langfuse 未装 / 初始化失败 → 返回 None。"""
    global _langfuse_handler, _langfuse_init_failed
    if _langfuse_handler is not None or _langfuse_init_failed:
        return _langfuse_handler
    if not _langfuse_enabled():
        return None
    try:
        # langfuse>=4 提供 langfuse.langchain.CallbackHandler,内部读 LANGFUSE_* env 自建 client
        from langfuse.langchain import CallbackHandler
        _langfuse_handler = CallbackHandler()
        logger.info(
            "[langfuse] callback enabled host=%s",
            os.getenv("LANGFUSE_HOST", "<default https://cloud.langfuse.com>"),
        )
    except Exception as e:
        _langfuse_init_failed = True
        logger.warning("[langfuse] handler init failed, tracing disabled: %s", e)
    return _langfuse_handler


def _inject_langfuse_callback(cfg: dict) -> dict:
    """如果 Langfuse 启用,把 handler 追加到 config['callbacks']。无 side-effect 友好,可重复调用。"""
    handler = _get_langfuse_handler()
    if handler is None:
        return cfg
    callbacks = list(cfg.get("callbacks") or [])
    if handler not in callbacks:
        callbacks.append(handler)
    cfg["callbacks"] = callbacks
    return cfg


# 启动时单次状态日志
if _langfuse_enabled():
    # 真正实例化推迟到第一次 invoke,这里只通报 env 已就绪
    logger.info(
        "[langfuse] env keys present, host=%s (handler lazy-init on first invoke)",
        os.getenv("LANGFUSE_HOST", "<default cloud>"),
    )
else:
    logger.info("[langfuse] env keys missing, skip (set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)")


def mask_key(k: str) -> str:
    """掩码 API Key，只保留前 6 后 4 便于排查又不泄漏。"""
    if not k:
        return "<empty>"
    if len(k) <= 10:
        return "***"
    return f"{k[:6]}...{k[-4:]}"


# _ssl_ctx 已在 runtime.py 统一构造,这里只 re-export(顶部 import 已 alias)

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

# ==================== Tools 已移到 tools/ 子包 ====================
# 9 个 @tool 函数(calculator/get_current_time/text_analyzer/unit_converter/
# word_counter/get_weather/search_knowledge_base/fetch_url/web_search)拆分到
# tools/ 目录,每个文件一个工具,统一从 `from tools import ALL_TOOLS` 引入。
# 共享运行时状态(知识库 / BGE 模型 / SSL ctx)在 runtime.py。
# main.py 顶部已 re-export 所有名字,测试和外部调用方仍可通过 main.<tool_name> 访问。


# uuid 用于 /knowledge-base /workflow /agent endpoint 生成 ID
import uuid

# TOOLS 直接复用 tools/__init__.py 的 ALL_TOOLS,保留 main.TOOLS 这个旧名字
# 兼容 tests 和 /chat /chat/stream /run_workflow 等所有引用方
TOOLS = ALL_TOOLS


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