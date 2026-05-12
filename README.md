# LangChain Agent

[![GitHub stars](https://img.shields.io/github/stars/yli769227-jpg/langchainAgent?style=social)](https://github.com/yli769227-jpg/langchainAgent/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yli769227-jpg/langchainAgent?style=social)](https://github.com/yli769227-jpg/langchainAgent/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C.svg?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-FF6B6B.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![DeepAgents](https://img.shields.io/badge/DeepAgents-0.5-FF9966.svg)](https://github.com/langchain-ai/deepagents)
[![Tests](https://img.shields.io/badge/tests-53_passed-brightgreen.svg)](./files/tests)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yli769227-jpg/langchainAgent/pulls)

基于 **LangChain 1.2 + LangGraph 1.1 + FastAPI** 的全栈智能 Agent。
**真接入** LangChain 状态机栈,工具用 `@tool` 装饰,schema 自动派生,
无任何手写 OpenAI 工具循环。

## 技术栈

| 层 | 技术 | 版本 |
|----|------|------|
| Agent 框架 | LangChain | `1.2.x` |
| 状态机 | LangGraph | `1.1.x` |
| LLM 接入 | langchain-openai / langchain-anthropic | `1.2.x` / `1.4.x` |
| 后端 | FastAPI | `0.115+` |
| 嵌入 | sentence-transformers (BGE-small-zh) | `3.x+` |
| 前端 | 原生 HTML/CSS/JS | 无构建 |
| 模型 | 任何 OpenAI 兼容 endpoint（中转/本地/真 API） | — |

## 架构

```
浏览器 (index.html, 原生)
    │
    │ HTTP / SSE
    ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                          │
│  ├── /chat            → run_agent()                          │
│  ├── /chat/stream     → astream_events (SSE: token/tool/done)│
│  ├── /tools           → 自动派生的 OpenAI schema             │
│  ├── /knowledge-base  → 文档增删改 + BGE 向量化              │
│  ├── /workflow        → 节点 DAG 执行引擎                    │
│  └── /agent           → 多智能体 / 自主规划                  │
│        │                                                    │
│        ▼                                                    │
│  langchain.agents.create_agent  (LangGraph)                 │
│   __start__ → model ⇄ tools → __end__                       │
└──────────┬──────────────┬───────────────────────────────────┘
           │              │
           ▼              ▼
   ChatOpenAI       9 个 @tool 函数
  (任意 base_url)   ─ calculator (AST 白名单)
                   ─ get_current_time
                   ─ text_analyzer
                   ─ unit_converter
                   ─ word_counter
                   ─ get_weather (Open-Meteo)
                   ─ search_knowledge_base (BGE 向量 + 关键词融合)
                   ─ fetch_url
                   ─ web_search (DuckDuckGo)
```

## 内置工具（9 个）

| 工具名 | 功能 | 示例 |
|--------|------|------|
| `calculator` | AST 白名单数学求值，禁 import / lambda / 属性访问 | `sqrt(16)`、`sin(0)` |
| `get_current_time` | 获取当前时间与日期（北京时间） | "今天星期几？" |
| `text_analyzer` | 字符 / 词 / 行 / 中文字符统计 | "分析这段文字..." |
| `unit_converter` | 长度（m/km/mile/ft/cm/mm）、重量（kg/lb/g/oz）、温度（°C/°F/K） | "100km → mile" |
| `word_counter` | 词频统计（不区分大小写） | "统计 'the' 出现次数" |
| `get_weather` | Open-Meteo 实时天气 + WMO 描述 | "北京天气" |
| `search_knowledge_base` | BGE-small-zh 向量检索 + 关键词融合（0.7 / 0.3） | KB 问答 |
| `fetch_url` | 抓取 HTTP(S)，HTML 自动剥标签，截断 8000 字符 | "读 https://..." |
| `web_search` | DuckDuckGo Instant Answer，无需 Key | "什么是 langchain" |

工具的 OpenAI schema 由 `@tool` 装饰器 + docstring 自动派生（见
`langchain_core.utils.function_calling.convert_to_openai_tool`），无重复维护。

## 快速启动

```bash
cd files

# 1. 创建 venv 并装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量（可选）
cp .env.example .env
# 编辑 .env，填入 AGENT_LOGIN_PASSWORD 等

# 3. 启动后端
python3 main.py
# 服务运行在 http://0.0.0.0:8000
```

打开 `index.html`（直接双击或 `python3 -m http.server` 都可），
在左侧填入 API Key 与 base_url（任何 OpenAI 兼容 endpoint），
点 **连接 & 测试** 即可。

## 测试

```bash
source .venv/bin/activate
pytest tests/ -v
```

测试覆盖：
- `test_tools.py`：每个工具的 happy path + 边界（中文 / unsupported / 缺失 KB）
- `test_calc_safety.py`：AST 白名单（拒 import / lambda / 属性访问 / 推导式 / 字符串 / 深嵌套）
- `test_agent_dispatch.py`：用 `GenericFakeChatModel` mock LLM，
  端到端验证 `run_agent` → 工具调度 → step 返回值

**测试不依赖任何真 API**。

## API 端点

| Method | 路径 | 说明 |
|--------|------|------|
| `GET`  | `/health` | 健康检查 |
| `POST` | `/auth/login` | 后端密码校验（`AGENT_LOGIN_PASSWORD`） |
| `GET`  | `/tools` | 列出所有工具及描述 |
| `POST` | `/chat` | 同步对话，返回 `(response, steps)` |
| `POST` | `/chat/stream` | SSE 流式对话，事件：`token` / `tool` / `done` |
| `DELETE` | `/chat/history/{session_id}` | 清空会话历史 |
| `POST/GET/DELETE` | `/knowledge-base` | 知识库 CRUD |
| `POST` | `/knowledge-base/{kb_id}/document` | 加文档（同步生成 BGE 向量） |
| `POST/GET/PUT/DELETE` | `/workflow` | 工作流 CRUD |
| `POST` | `/workflow/run` | 执行工作流 |
| `POST/GET/PUT/DELETE` | `/agent` | Agent 编排 CRUD |
| `POST` | `/agent/{agent_id}/chat` | 多智能体 / 自主规划对话 |

### SSE 流式协议

`POST /chat/stream` 推送事件：

```
data: {"type":"token","content":"答"}
data: {"type":"token","content":"案"}
data: {"type":"tool","step":{"tool":"calculator","input":{"expression":"1+1"},"output":"= 2"}}
data: {"type":"done","steps":[...]}
```

## 环境变量

`.env.example`：

```
AGENT_LOGIN_PASSWORD=agent2024     # 前端登录密码（默认 agent2024）
AGENT_INSECURE_SSL=0               # 1 = 关闭证书校验（自签证书中转用）
```

API Key / base_url 从前端请求体传入，**服务器不持久化任何凭证**。

## 扩展工具

```python
from langchain_core.tools import tool

@tool
def my_tool(arg1: str, arg2: int = 10) -> str:
    """工具的描述。LangChain 用这个 docstring 派生 schema。

    Args:
        arg1: 参数 1 描述
        arg2: 参数 2 描述
    """
    return f"result: {arg1}/{arg2}"

# 加入 main.py 的 TOOLS 列表即可
TOOLS = [..., my_tool]
```

无需手写 OpenAI schema、无需更新 `TOOL_FUNCTIONS` dict——`TOOLS_SCHEMA` 与
`TOOL_FUNCTIONS` 都从 `TOOLS` 自动派生（兼容旧调用方）。

## 设计抉择

- **为什么不直接 OpenAI SDK？** 对接外部中转/换模型/换 provider 时，
  原本的手写循环要重写状态机；LangGraph 的 `model ⇄ tools` 状态机解耦了
  这部分逻辑。
- **为什么 ChatOpenAI 而非 OpenAI 直连？** `langchain-openai.ChatOpenAI`
  支持任意 `base_url`，覆盖了 99% 的中转 / 本地推理 / 真 API 场景，
  零额外代码。
- **为什么保留 `TOOLS_SCHEMA` / `TOOL_FUNCTIONS` 兼容字段？** workflow / agent
  模块的旧调用路径用了它们；改成自动派生但接口不动，最小化破坏面。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yli769227-jpg/langchainAgent&type=Date)](https://star-history.com/#yli769227-jpg/langchainAgent&Date)

## License

MIT — see [LICENSE](../LICENSE).

## Contributing

PR / Issue 欢迎。提 PR 前请:
1. 跑通 `pytest tests/`(目前 40 个测试)
2. 加测试覆盖你的新工具 / 新功能
3. 在 README "工具" 表里登记新工具
