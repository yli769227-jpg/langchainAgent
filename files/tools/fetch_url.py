"""fetch_url 工具:抓取 HTTP(S) URL 正文,HTML 自动剥标签。"""

import logging
import re
import urllib.request

from langchain_core.tools import tool

try:
    from runtime import _ssl_ctx
except ImportError:  # 当 tools 作为 files.tools 子包加载时(repo 根 import)
    from ..runtime import _ssl_ctx  # type: ignore

logger = logging.getLogger("agent.tools.fetch_url")


@tool
def fetch_url(url: str, max_chars: int = 2000) -> str:
    """抓取任意 HTTP(S) URL 的正文内容。HTML 会自动剥标签并压缩空白；JSON/纯文本原样返回。超过 max_chars 会截断。用于读文章/API/在线文档。

    Args:
        url: 目标 URL，必须以 http:// 或 https:// 开头
        max_chars: 返回正文最大字符数，默认 2000，上限 8000
    """
    if not re.match(r"^https?://", url):
        return "url 必须以 http:// 或 https:// 开头"
    if max_chars > 8000:
        max_chars = 8000
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AgentFlow)"})
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(1024 * 1024)  # 最多 1MB
            body = raw.decode(resp.headers.get_content_charset() or "utf-8", errors="replace")
        # 简易 HTML→文本:去 script/style 再剥标签
        if "html" in ctype.lower() or body.lstrip().startswith("<"):
            body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body, flags=re.DOTALL | re.IGNORECASE)
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
        truncated = len(body) > max_chars
        body = body[:max_chars]
        return f"URL: {url}\nContent-Type: {ctype}\n\n{body}" + ("\n\n[... 已截断]" if truncated else "")
    except Exception as e:
        logger.warning("[fetch_url] 抓取失败 url=%s err=%s", url, e)
        return f"抓取失败: {type(e).__name__}: {str(e)}"
