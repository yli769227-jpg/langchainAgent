"""共享运行时状态 (shared runtime state).

放置被多个 tool / endpoint 共用的全局对象,避免 tools 子模块和 main.py 循环依赖。

- knowledge_bases: 内存知识库 dict,/knowledge-base API 和 search_knowledge_base tool 共用
- _ssl_ctx: HTTP(S) 抓取共用的 SSLContext (受 AGENT_INSECURE_SSL 控制)
- BGE 模型懒加载 + 向量/关键词检索辅助函数 (search_knowledge_base + add_document API 共用)

main.py 顶部 `from runtime import knowledge_bases, _ssl_ctx, _get_bge_model, _encode`
保持原 main.* 命名兼容,测试和 /knowledge-base API 不需要改动。
"""

import logging
import os
import ssl

logger = logging.getLogger("agent.runtime")

# 第三方中转服务可能用自签证书,默认仍校验;由环境变量开启关闭
_ssl_ctx = ssl.create_default_context()
if os.getenv("AGENT_INSECURE_SSL") == "1":
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE
    logger.warning("[runtime] SSL 证书验证已关闭 (AGENT_INSECURE_SSL=1)")

# 内存存储:{kb_id: {"name": str, "description": str, "docs": [str], "vectors": list}}
knowledge_bases: dict[str, dict] = {}

# BGE 模型懒加载
_bge_model = None


def _get_bge_model():
    """懒加载 BGE-small-zh 模型;加载失败返回 None,调用方降级到纯关键词。"""
    global _bge_model
    if _bge_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[runtime] 首次加载 BGE 模型 BAAI/bge-small-zh-v1.5")
            _bge_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except Exception as e:
            logger.warning("[runtime] BGE 模型加载失败,降级为纯关键词检索: %s", e)
            _bge_model = False  # 标记加载失败
    return _bge_model if _bge_model is not False else None


def _keyword_score(doc: str, query: str) -> float:
    """简易关键词匹配打分:词匹配 + 中文字符匹配。"""
    query_words = set(query.lower().split())
    doc_lower = doc.lower()
    score = sum(1 for w in query_words if w in doc_lower)
    score += sum(1 for c in query if c.strip() and c in doc)
    return float(score)


def _encode(texts: list) -> list:
    """用 BGE 模型编码并归一化;模型不可用时返回空列表。"""
    model = _get_bge_model()
    if model is None:
        return []
    return model.encode(texts, normalize_embeddings=True).tolist()


def _cosine(a: list, b: list) -> float:
    """已 normalize 的向量,点积 == cosine。"""
    return sum(x * y for x, y in zip(a, b))
