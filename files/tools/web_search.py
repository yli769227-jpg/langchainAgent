"""web_search 工具:用 DuckDuckGo Instant Answer API 做免 Key 搜索。"""

import json
import logging
import urllib.parse
import urllib.request

from langchain_core.tools import tool

try:
    from runtime import _ssl_ctx
except ImportError:  # 当 tools 作为 files.tools 子包加载时(repo 根 import)
    from ..runtime import _ssl_ctx  # type: ignore

logger = logging.getLogger("agent.tools.web_search")


@tool
def web_search(query: str) -> str:
    """用 DuckDuckGo 搜索网络信息，返回摘要和相关主题。适合快速查概念/时事/知识问答，无需 API Key。若需完整正文请用 fetch_url。

    Args:
        query: 搜索关键词或自然语言问题
    """
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
        req = urllib.request.Request(
            f"https://api.duckduckgo.com/?{params}",
            headers={"User-Agent": "Mozilla/5.0 (AgentFlow)"},
        )
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
        logger.warning("[web_search] 搜索失败 query=%s err=%s", query, e)
        return f"搜索失败: {type(e).__name__}: {str(e)}"
