"""word_counter 工具:统计特定词语在文本中出现的次数(不区分大小写)。"""

from langchain_core.tools import tool


@tool
def word_counter(text: str, target_word: str) -> str:
    """在文本中统计特定词语出现的次数（不区分大小写）。

    Args:
        text: 要搜索的文本
        target_word: 要统计的词语
    """
    count = text.lower().count(target_word.lower())
    return f"词语 '{target_word}' 在文本中出现了 {count} 次"
