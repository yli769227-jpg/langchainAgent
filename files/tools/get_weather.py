"""get_weather 工具:用 Open-Meteo 免费 API 查实时天气。"""

import json
import logging
import urllib.parse
import urllib.request

from langchain_core.tools import tool

try:
    from runtime import _ssl_ctx
except ImportError:  # 当 tools 作为 files.tools 子包加载时(repo 根 import)
    from ..runtime import _ssl_ctx  # type: ignore

logger = logging.getLogger("agent.tools.weather")


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，包括温度、湿度、风速、天气状况等。

    Args:
        city: 城市名称，支持中文或英文，如 '北京'、'Shanghai'、'London'
    """
    try:
        # 第一步:用 open-meteo geocoding API 将城市名转为经纬度
        encoded_city = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=zh&format=json"
        req = urllib.request.Request(geo_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            geo_data = json.loads(resp.read().decode())

        if not geo_data.get("results"):
            return f"找不到城市: {city}"

        result = geo_data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        city_name = result.get("name", city)
        country = result.get("country", "")

        # 第二步:用经纬度查询实时天气
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"weather_code,wind_speed_10m,wind_direction_10m,visibility"
            f"&wind_speed_unit=kmh&timezone=auto"
        )
        req2 = urllib.request.Request(weather_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=10, context=_ssl_ctx) as resp2:
            w_data = json.loads(resp2.read().decode())

        cur = w_data["current"]
        temp = cur["temperature_2m"]
        feels_like = cur["apparent_temperature"]
        humidity = cur["relative_humidity_2m"]
        wind_speed = cur["wind_speed_10m"]
        wind_dir = cur["wind_direction_10m"]
        visibility = cur.get("visibility", "N/A")
        wmo_code = cur["weather_code"]

        # WMO 天气代码简单映射
        wmo_desc = {
            0: "晴天", 1: "基本晴朗", 2: "局部多云", 3: "阴天",
            45: "雾", 48: "冻雾",
            51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
            95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
        }
        desc = wmo_desc.get(wmo_code, f"天气代码 {wmo_code}")

        vis_str = f"{int(visibility/1000)} km" if isinstance(visibility, (int, float)) else str(visibility)

        return (
            f"🌤 {city_name}, {country} 实时天气\n"
            f"  天气状况: {desc}\n"
            f"  温度: {temp}°C（体感 {feels_like}°C）\n"
            f"  湿度: {humidity}%\n"
            f"  风速: {wind_speed} km/h，风向: {wind_dir}°\n"
            f"  能见度: {vis_str}"
        )
    except Exception as e:
        logger.warning("[get_weather] 查询失败 city=%s err=%s", city, e)
        return f"天气查询失败: {str(e)}"
