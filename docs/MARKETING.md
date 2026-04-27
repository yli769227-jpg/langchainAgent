# 推广模板 (Marketing Templates)

收集站外引流文案 / 提交 awesome-list 模板,自取自用。仓库: <https://github.com/yli769227-jpg/langchainAgent>

---

## 1. X (Twitter) 推文

### 英文

```
Just open-sourced langchainAgent — a full-stack AI agent built on LangChain 1.2 + LangGraph 1.1 + FastAPI.

✨ 9 built-in tools (calc, web, units, RAG)
🇨🇳 Chinese knowledge base via BGE embeddings
🔄 Multi-agent workflow orchestration
⚡ SSE streaming, vanilla JS frontend (no build)
✅ 40 tests, MIT licensed

→ github.com/yli769227-jpg/langchainAgent

#LangChain #LangGraph #LLM #AIagent #Python
```

### 中文

```
开源了一个 LangChain Agent 框架 —— 真正用 LangChain 1.2 + LangGraph 1.1,不是挂羊头卖狗肉那种 🙃

✨ 9 个内置工具(计算/搜网/单位/RAG)
🇨🇳 中文知识库 BGE 嵌入开箱即用
🔄 多 Agent 工作流编排
⚡ SSE 流式,原生 JS 前端无构建
✅ 40 个测试,MIT 协议

→ github.com/yli769227-jpg/langchainAgent

求 Star ⭐ #LangChain #LLM #AIagent
```

---

## 2. 小红书 / 微博 长文

### 标题候选
- "我把 LangChain Agent 项目从『挂羊头卖狗肉』改成『名实合一』"
- "开源:基于 LangChain + LangGraph 的中文 AI Agent 框架"
- "全栈 LLM Agent 模板:9 工具 + RAG + 工作流 + SSE 流式,40 测试"

### 正文骨架

```
最近发现一个有意思的现象:很多叫做 langchainAgent 的项目其实根本没有 import langchain,
都是手写 OpenAI 工具循环。我自己的项目也犯过这毛病 —— 用了 60 行代码手撸 ReAct 循环,
README 还说"基于 LangChain"。

最近彻底重写了一遍,把它做成真正的 LangChain Agent。

【最大的几个变化】
1. 60 行手写循环 → langchain.agents.create_agent (一行调用)
2. 140 行手写 OpenAI tool schema → @tool 装饰器,自动派生
3. 0 测试 → 40 测试(工具 / AST 安全 / dispatch mock)
4. README 撒谎 → 实事求是 + 架构图

【为什么用 LangGraph?】
看似过度工程,实际有用:
- 换模型 / 换 provider 时,状态机不变
- ChatOpenAI 支持任意 base_url,覆盖中转 / 本地 / 真 API 三个场景零改代码
- 流式输出走 astream_events,不用自己解析 chunk

【特色】
- 中文向量化用的是 BAAI/bge-small-zh-v1.5,中文语义召回比 OpenAI ada 好不少
- 工作流支持 sequential / parallel / debate / planner-executor-reflector 四种模式
- 前端原生 HTML/CSS/JS,不需要 npm install

GitHub: github.com/yli769227-jpg/langchainAgent
欢迎拍砖 + Star ⭐
```

---

## 3. Reddit r/LangChain 帖子

### 标题
```
[Showcase] langchainAgent — full-stack LangChain 1.2 + LangGraph 1.1 + FastAPI agent with 9 tools, RAG, multi-agent workflows
```

### 正文

```
Hi r/LangChain,

I've been working on a full-stack agent template that actually uses LangChain (yes, ironic that I had to clarify this — my earlier version was literally just OpenAI SDK calls). Just shipped the migration:

**Stack**: LangChain 1.2.15 / LangGraph 1.1.9 / FastAPI / vanilla JS frontend

**Highlights**:
- 9 @tool decorated functions (calc with AST safety whitelist, time, text analysis, units, web fetch, web search, RAG)
- Chinese knowledge base built on BAAI/bge-small-zh-v1.5 (significantly better Chinese recall than OpenAI ada)
- Workflow engine: sequential / parallel / debate / planner-executor-reflector modes
- /chat/stream uses astream_events(version="v2"), preserves SSE token/tool/done event protocol
- 40 pytest cases (tools, AST safety, agent dispatch with mocked LLM)
- MIT licensed, no JS build step required

**The migration commit**:
- 60 LOC handwritten OpenAI tool-loop → 1 call to `langchain.agents.create_agent`
- 140 LOC manual TOOLS_SCHEMA dict → derived from @tool decorators
- net -388 / +760 (mostly tests + README)

Source: github.com/yli769227-jpg/langchainAgent

Feedback welcome, especially on:
- Whether `astream_events` is the idiomatic way to surface tool steps to a UI
- LangGraph workflow patterns I should add (currently only the 4 prebuilt modes)

Stars appreciated 🙏
```

---

## 4. LangChain 中文社区(微信群/Discord/飞书群)模板

```
【开源】LangChain Agent 中文优化版

刚把项目从手写 OpenAI 循环迁移到真正的 LangChain 1.2 + LangGraph 1.1。
特别针对中文场景做了优化:
- 知识库默认 BGE-zh-small,中文语义比 OpenAI ada 强
- README 全中文 + 架构图
- 9 个开箱即用工具
- 工作流 4 种模式(顺序/并行/辩论/计划-执行-反思)

GitHub:github.com/yli769227-jpg/langchainAgent

求 Star,有问题欢迎 issue 拍砖
```

---

## 5. Awesome-list 提交模板

向 [awesome-langchain](https://github.com/kyrolabs/awesome-langchain) /
[awesome-llm-agents](https://github.com/Charmve/awesome-llm-agents) 提 PR
时,在合适位置加一行:

```markdown
- [langchainAgent](https://github.com/yli769227-jpg/langchainAgent) — Full-stack agent (LangChain 1.2 + LangGraph 1.1 + FastAPI) with 9 tools, BGE Chinese embeddings RAG, multi-agent workflow modes, SSE streaming. Vanilla JS frontend.
```

---

## 6. Hacker News 提交标题

```
Show HN: langchainAgent — Full-stack LangChain agent with RAG, workflows, SSE streaming
```

URL 字段填:`https://github.com/yli769227-jpg/langchainAgent`

---

## 7. ProductHunt 描述

**Tagline (60 字符内)**:
```
Open-source full-stack LangChain agent with RAG and workflows
```

**Description**:
```
Production-ready AI agent template built on LangChain 1.2 + LangGraph 1.1 + FastAPI.
Comes with 9 built-in tools (calc, web, RAG), Chinese knowledge base via BGE embeddings,
multi-agent workflow orchestration (sequential/parallel/debate/planner), SSE streaming,
and a vanilla JS frontend that works without npm. MIT licensed, 40 tests, ready to fork.
```

---

## 8. 投放节奏建议

| 阶段 | 渠道 | 时机 |
|---|---|---|
| Day 1 | X / 微博 / 小红书 | 同时发,@langchain 官方账号 |
| Day 2 | r/LangChain | 等 Day 1 反馈再投,根据互动调整文案 |
| Day 3 | LangChain 中文社区微信群 / Discord | 多个群分时段发 |
| Day 7 | Hacker News Show HN | 上午 8-10 点 PT(美国早晨)效果最好 |
| Day 14 | Awesome-list PR | 等项目有 50+ Star 后再提,通过率高 |
| Day 30 | ProductHunt | 项目至少 100 Star + 1 issue + 1 fork 时再投 |

> **诚实原则**: 文案里的"40 测试""真接入 LangChain""9 工具"这些数字是事实,
> 不要为了流量编。审核被拒一次后续传播都会受影响。
