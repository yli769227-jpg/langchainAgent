"""search_knowledge_base 工具:在内存知识库中做向量+关键词融合检索。"""

import logging

from langchain_core.tools import tool

try:
    from runtime import (
        _cosine,
        _encode,
        _get_bge_model,
        _keyword_score,
        knowledge_bases,
    )
except ImportError:  # 当 tools 作为 files.tools 子包加载时(repo 根 import)
    from ..runtime import (  # type: ignore
        _cosine,
        _encode,
        _get_bge_model,
        _keyword_score,
        knowledge_bases,
    )

logger = logging.getLogger("agent.tools.kb")


@tool
def search_knowledge_base(kb_id: str, query: str, top_k: int = 3) -> str:
    """在指定知识库中搜索相关内容，用于回答基于知识库的问题。

    Args:
        kb_id: 知识库 ID
        query: 搜索查询内容
        top_k: 返回最相关的文档数量，默认 3
    """
    if kb_id not in knowledge_bases:
        return f"知识库 {kb_id} 不存在"
    kb = knowledge_bases[kb_id]
    docs = kb["docs"]
    if not docs:
        return "知识库为空"

    vectors = kb.get("vectors", [])
    model = _get_bge_model()

    scored = []
    if model and len(vectors) == len(docs):
        # 向量检索
        query_vec = _encode(["为这个句子生成表示以用于检索相关文章：" + query])[0]
        for i, doc in enumerate(docs):
            vec_score = _cosine(query_vec, vectors[i])
            kw_score = _keyword_score(doc, query)
            # 归一化关键词分数后融合,向量权重 0.7,关键词权重 0.3
            max_kw = max((_keyword_score(d, query) for d in docs), default=1) or 1
            combined = 0.7 * vec_score + 0.3 * (kw_score / max_kw)
            scored.append((combined, doc))
    else:
        # 降级:纯关键词
        for doc in docs:
            scored.append((_keyword_score(doc, query), doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [doc for score, doc in scored[:top_k] if score > 0]
    if not results:
        return "未找到相关内容"
    return "\n\n---\n\n".join(results)
