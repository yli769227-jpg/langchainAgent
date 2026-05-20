"""get_current_time 工具:返回当前北京时间。"""

import datetime

from langchain_core.tools import tool


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前日期和时间（北京时间）。

    Args:
        timezone: 时区，默认 Asia/Shanghai
    """
    now = datetime.datetime.now()
    return (
        f"当前时间（北京时间）: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期: {['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}\n"
        f"今年第 {now.timetuple().tm_yday} 天"
    )
