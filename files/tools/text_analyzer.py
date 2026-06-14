"""text_analyzer 工具:文本统计(字符/词/行/中文字符)。"""

from langchain_core.tools import tool


@tool
def text_analyzer(text: str) -> str:
    """分析文本统计信息：字符数、词数、行数、中文字符数等。

    Args:
        text: 要分析的文本
    """
    lines = text.split('\n')
    words = text.split()
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    return (
        f"📊 文本分析结果:\n"
        f"  - 总字符数: {len(text)}\n"
        f"  - 中文字符数: {chinese_chars}\n"
        f"  - 英文单词数: {len(words)}\n"
        f"  - 行数: {len(lines)}\n"
        f"  - 段落数: {len([l for l in lines if l.strip()])}"
    )
