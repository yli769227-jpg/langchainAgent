"""unit_converter 工具:长度/重量/温度的常见单位换算。"""

from langchain_core.tools import tool


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。支持长度(m,km,mile,ft,cm,mm)、重量(kg,lb,g,oz)、温度(celsius,fahrenheit,kelvin)。

    Args:
        value: 要换算的数值
        from_unit: 原单位
        to_unit: 目标单位
    """
    conversions = {
        ("m", "km"): 0.001,         ("km", "m"): 1000,
        ("m", "mile"): 0.000621371, ("mile", "m"): 1609.34,
        ("m", "ft"): 3.28084,       ("ft", "m"): 0.3048,
        ("m", "cm"): 100,           ("cm", "m"): 0.01,
        ("m", "mm"): 1000,          ("mm", "m"): 0.001,
        ("km", "mile"): 0.621371,   ("mile", "km"): 1.60934,
        ("cm", "mm"): 10,           ("mm", "cm"): 0.1,
        ("kg", "lb"): 2.20462,      ("lb", "kg"): 0.453592,
        ("kg", "g"): 1000,          ("g", "kg"): 0.001,
        ("kg", "oz"): 35.274,       ("oz", "kg"): 0.0283495,
        ("g", "oz"): 0.035274,      ("oz", "g"): 28.3495,
    }
    key = (from_unit.lower(), to_unit.lower())
    f, t = from_unit.lower(), to_unit.lower()
    if f == "celsius" and t == "fahrenheit":
        result = value * 9/5 + 32
    elif f == "fahrenheit" and t == "celsius":
        result = (value - 32) * 5/9
    elif f == "celsius" and t == "kelvin":
        result = value + 273.15
    elif f == "kelvin" and t == "celsius":
        result = value - 273.15
    elif key in conversions:
        result = value * conversions[key]
    else:
        return f"不支持 {from_unit} 到 {to_unit} 的换算"
    return f"{value} {from_unit} = {result:.4f} {to_unit}"
