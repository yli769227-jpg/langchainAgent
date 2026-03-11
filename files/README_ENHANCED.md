# LangChain Agent Enhanced 🚀

增强版智能 Agent 系统，新增**记忆系统**、**知识库**和**工作流**功能。

## 🆕 新增功能

### 1. 记忆系统（Memory System）
- **短期记忆**：保留最近 20 条对话
- **长期记忆**：基于向量数据库的语义检索
- **智能召回**：自动检索相关历史对话
- **重要性评分**：对关键信息进行优先存储

### 2. 知识库（Knowledge Base）
- **RAG 集成**：检索增强生成
- **文档上传**：支持 TXT、PDF、DOCX 等格式
- **语义搜索**：基于向量相似度的智能检索
- **多知识库**：支持创建和管理多个知识库

### 3. 工作流系统（Workflow）
- **节点类型**：LLM、工具、条件、记忆、知识库、合并
- **条件分支**：支持 if-else 逻辑
- **可视化**：工作流图形化展示（前端待实现）
- **灵活编排**：自定义执行流程

## 快速开始

### 1. 安装依赖

```bash
cd /Users/zhangyida/Documents/agent/files
pip install -r requirements_enhanced.txt
```

**注意**：如果不需要向量存储功能，可以跳过 `chromadb` 和 `sentence-transformers`，系统会自动降级到简单记忆模式。

### 2. 启动增强版后端

```bash
python main_enhanced.py
```

启动后会显示：
```
🚀 LangChain Agent Enhanced Starting...
   Vector Store: ✅ Enabled
   Memory System: ✅ Enabled
   Knowledge Base: ✅ Enabled
```

### 3. 使用原有前端

直接用浏览器打开 `index.html`，后端 API 完全兼容原版。

## API 文档

### 基础对话

```bash
POST /chat
{
  "message": "你好",
  "session_id": "user123",
  "api_key": "sk-...",
  "knowledge_base_ids": ["kb_001"]  # 可选：指定知识库
}
```

### 知识库管理

#### 创建知识库
```bash
POST /knowledge-base?name=产品手册&description=公司产品文档
```

#### 上传文档
```bash
POST /knowledge-base/{kb_id}/upload
Content-Type: multipart/form-data
file: document.txt
```

#### 添加文本
```bash
POST /knowledge-base/{kb_id}/document
{
  "content": "这是一段知识库内容...",
  "metadata": {"source": "manual", "page": 1}
}
```

#### 列出知识库
```bash
GET /knowledge-base
```

### 记忆管理

#### 查看会话记忆
```bash
GET /memory/{session_id}?limit=10
```

#### 搜索记忆
```bash
POST /memory/{session_id}/search
{
  "query": "上次讨论的主题",
  "top_k": 5
}
```

### 工作流管理

#### 创建工作流
```bash
POST /workflow
{
  "id": "wf_001",
  "name": "客服工作流",
  "start_node": "node_1",
  "nodes": [
    {
      "id": "node_1",
      "type": "condition",
      "config": {"condition": "intent == 'query'"},
      "next_nodes": ["node_2", "node_3"]
    },
    {
      "id": "node_2",
      "type": "knowledge",
      "config": {"kb_ids": ["kb_001"]},
      "next_nodes": ["node_4"]
    }
  ]
}
```

#### 列出工作流
```bash
GET /workflow
```

## 使用示例

### 示例 1：使用记忆功能

```python
# 第一次对话
POST /chat
{
  "message": "我喜欢吃苹果",
  "session_id": "user123",
  "api_key": "sk-..."
}

# 后续对话（Agent 会自动检索相关记忆）
POST /chat
{
  "message": "我之前说过喜欢吃什么水果？",
  "session_id": "user123",
  "api_key": "sk-..."
}
# 响应：根据记忆检索，你之前提到喜欢吃苹果。
```

### 示例 2：使用知识库

```python
# 1. 创建知识库
kb_id = create_knowledge_base("产品FAQ", "常见问题解答")

# 2. 添加文档
add_document_to_kb(kb_id, """
Q: 如何重置密码？
A: 点击登录页面的"忘记密码"，输入邮箱接收重置链接。
""")

# 3. 对话时使用知识库
POST /chat
{
  "message": "怎么重置密码？",
  "session_id": "user123",
  "api_key": "sk-...",
  "knowledge_base_ids": [kb_id]
}
# Agent 会自动从知识库检索相关信息并回答
```

### 示例 3：工作流（预留功能）

```python
# 定义客服工作流
workflow = {
  "id": "customer_service",
  "name": "智能客服",
  "start_node": "intent_detection",
  "nodes": [
    {
      "id": "intent_detection",
      "type": "llm",
      "config": {"prompt": "判断用户意图：查询/投诉/建议"},
      "next_nodes": ["route"]
    },
    {
      "id": "route",
      "type": "condition",
      "condition": "intent == '查询'",
      "next_nodes": ["knowledge_search", "human_agent"]
    },
    {
      "id": "knowledge_search",
      "type": "knowledge",
      "config": {"kb_ids": ["faq_kb"]},
      "next_nodes": ["response"]
    }
  ]
}
```

## 工具列表

| 工具名 | 功能 | 新增 |
|--------|------|------|
| `calculator` | 数学计算 | ❌ |
| `get_current_time` | 获取时间 | ❌ |
| `text_analyzer` | 文本分析 | ❌ |
| `unit_converter` | 单位换算 | ❌ |
| `word_counter` | 词频统计 | ❌ |
| `search_memory` | 搜索历史记忆 | ✅ |
| `query_knowledge_base` | 查询知识库 | ✅ |

## 技术架构

```
┌─────────────────────────────────────────┐
│           Frontend (index.html)         │
│         原有前端界面（完全兼容）           │
└─────────────────┬───────────────────────┘
                  │ HTTP/REST API
┌─────────────────▼───────────────────────┐
│        FastAPI Backend (Enhanced)       │
│  ┌─────────────────────────────────┐   │
│  │      Agent Execution Engine      │   │
│  │  - Tool calling loop             │   │
│  │  - Memory integration            │   │
│  │  - Knowledge retrieval           │   │
│  └─────────────────────────────────┘   │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Memory   │  │Knowledge │  │Workflow││
│  │ System   │  │   Base   │  │ Engine ││
│  └────┬─────┘  └────┬─────┘  └────────┘│
└───────┼─────────────┼──────────────────┘
        │             │
┌───────▼─────────────▼──────────────────┐
│       ChromaDB (Vector Store)          │
│  - Semantic search                     │
│  - Embedding storage                   │
└────────────────────────────────────────┘
```

## 配置说明

### 环境变量（可选）

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_MODEL=gpt-4o-mini
```

### 向量数据库

默认使用 ChromaDB 的内存模式，数据不持久化。如需持久化：

```python
# 修改 main_enhanced.py
chroma_client = ChromaClient(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))
```

## 扩展开发

### 添加新工具

```python
def my_custom_tool(param: str) -> str:
    """工具描述"""
    return f"结果: {param}"

# 注册工具
TOOL_FUNCTIONS["my_custom_tool"] = my_custom_tool

# 添加 Schema
TOOLS_SCHEMA.append({
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "工具描述",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "参数说明"}
            },
            "required": ["param"]
        }
    }
})
```

### 自定义知识库模板

```python
# 示例：创建产品文档知识库
def setup_product_kb():
    kb_id = create_knowledge_base(
        name="产品文档",
        description="公司所有产品的使用手册和 FAQ"
    )

    # 批量导入文档
    documents = [
        "产品 A 使用说明...",
        "产品 B 常见问题...",
        "产品 C 技术规格..."
    ]

    for doc in documents:
        add_document_to_kb(kb_id, doc)

    return kb_id
```

## 性能优化建议

1. **向量数据库**：生产环境建议使用 Pinecone 或 Weaviate
2. **缓存**：对频繁查询的知识库结果进行缓存
3. **异步处理**：大文件上传使用后台任务
4. **分块策略**：根据文档类型调整分块大小（当前 500 字符）

## 下一步计划

- [ ] 前端可视化工作流编辑器
- [ ] 支持更多文档格式（PDF、DOCX、Markdown）
- [ ] 记忆重要性自动评分（基于 LLM）
- [ ] 多模态支持（图片、音频）
- [ ] 工作流执行引擎完整实现
- [ ] 知识库版本管理

## 常见问题

**Q: 不安装 ChromaDB 可以运行吗？**
A: 可以。系统会自动降级到简单记忆模式，但无法使用语义搜索功能。

**Q: 如何切换到其他向量数据库？**
A: 修改 `main_enhanced.py` 中的向量存储初始化代码，实现相同的接口即可。

**Q: 知识库支持哪些文件格式？**
A: 当前支持纯文本。安装 `pypdf` 和 `python-docx` 后可支持 PDF 和 Word。

**Q: 记忆会永久保存吗？**
A: 默认使用内存模式，重启后丢失。配置持久化目录可永久保存。

## 技术栈

- **FastAPI** - 后端框架
- **OpenAI API** - LLM 接口
- **ChromaDB** - 向量数据库
- **Sentence Transformers** - 文本嵌入
- **Pydantic** - 数据验证

## License

MIT
