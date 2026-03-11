# LangChain Agent 🤖

基于 **LangChain + FastAPI + 精美前端** 的全栈智能 Agent 项目。

## 项目结构

```
langchain-agent/
├── backend/
│   ├── main.py              # FastAPI 后端 + LangChain Agent
│   ├── requirements.txt     # Python 依赖
│   └── .env.example         # 环境变量模板
└── frontend/
    └── index.html           # 前端界面（无需构建，直接运行）
```

## 内置工具

| 工具名 | 功能 | 示例用法 |
|--------|------|---------|
| `calculator` | 数学计算（支持三角函数、对数等） | "计算 sin(π/4)" |
| `get_current_time` | 获取当前时间和日期 | "今天星期几？" |
| `text_analyzer` | 文本字符/词数/行数统计 | "分析这段文字..." |
| `unit_converter` | 长度、重量、温度单位换算 | "100公里是多少英里" |
| `word_counter` | 词语频率统计 | "统计'的'出现次数" |

## 快速启动

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动后端

```bash
cd backend
python main.py
# 服务运行在 http://localhost:8000
```

### 3. 打开前端

直接用浏览器打开 `frontend/index.html`，**无需任何构建**。

### 4. 配置连接

在前端界面左侧侧边栏：
1. 确认后端地址为 `http://localhost:8000`
2. 输入你的 OpenAI API Key（格式：`sk-...`）
3. 点击 **连接 & 测试**

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/tools` | GET | 获取工具列表 |
| `/chat` | POST | 发送消息 |
| `/chat/{session_id}` | DELETE | 清除会话历史 |

### Chat 请求示例

```json
POST /chat
{
  "message": "100公里等于多少英里？",
  "session_id": "my_session",
  "api_key": "sk-your-key"
}
```

## 扩展工具

在 `backend/main.py` 中添加新工具，使用 `@tool` 装饰器：

```python
@tool
def my_new_tool(param: str) -> str:
    """工具描述，LLM 根据这个描述决定何时调用此工具"""
    # 实现逻辑
    return f"结果: {param}"
```

然后将工具添加到 `tools` 列表即可。

## 技术栈

- **LangChain** `0.3.x` - Agent 框架
- **FastAPI** - 后端 API
- **OpenAI GPT-4o-mini** - LLM（可替换为其他模型）
- **纯 HTML/CSS/JS** - 前端（零依赖）
