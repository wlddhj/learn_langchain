"""
第9章 Demo 1：@tool 装饰器与 StructuredTool

演示 @tool 装饰器、类型注解、StructuredTool、args_schema。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 1. @tool 装饰器基础用法
# ============================================================

def demo_basic_tool():
    print("=" * 50)
    print("Demo 9-1 (1/3): @tool 装饰器")
    print("=" * 50)

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气信息"""
        weather_data = {
            "北京": "晴天，25°C，湿度 45%",
            "上海": "多云，22°C，湿度 65%",
            "深圳": "小雨，28°C，湿度 80%",
            "成都": "阴天，20°C，湿度 70%",
        }
        return weather_data.get(city, f"未找到 {city} 的天气信息")

    # 查看工具属性
    print(f"名称: {get_weather.name}")
    print(f"描述: {get_weather.description}")
    print(f"参数: {get_weather.args}")
    print()

    # 直接调用
    print(f"调用 get_weather('北京'): {get_weather.invoke({'city': '北京'})}")
    print(f"调用 get_weather('东京'): {get_weather.invoke({'city': '东京'})}")
    print()

    # 多参数工具
    @tool
    def search_products(
        keyword: str,
        category: Optional[str] = None,
        max_price: float = 1000.0,
    ) -> str:
        """搜索商品信息。
        keyword: 搜索关键词
        category: 商品分类（可选）
        max_price: 最高价格"""
        results = f"搜索 '{keyword}'"
        if category:
            results += f"，分类: {category}"
        results += f"，最高价: ¥{max_price}"
        return results

    print(f"参数 schema: {search_products.args_schema.model_json_schema()}")
    print(f"调用: {search_products.invoke({'keyword': '手机', 'category': '数码', 'max_price': 5000})}")
    print()


# ============================================================
# 2. StructuredTool 精细控制
# ============================================================

def demo_structured_tool():
    print("=" * 50)
    print("Demo 9-1 (2/3): StructuredTool")
    print("=" * 50)

    class CalculatorInput(BaseModel):
        a: float = Field(description="第一个数字")
        b: float = Field(description="第二个数字")
        operation: str = Field(description="运算类型: add/subtract/multiply/divide")

    def calculator_func(a: float, b: float, operation: str) -> str:
        ops = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "错误: 除数不能为零",
        }
        if operation not in ops:
            return f"不支持的运算: {operation}，支持: add/subtract/multiply/divide"
        result = ops[operation](a, b)
        return f"{a} {operation} {b} = {result}"

    calculator = StructuredTool.from_function(
        func=calculator_func,
        name="calculator",
        description="执行基本数学运算。支持加(add)、减(subtract)、乘(multiply)、除(divide)。",
        args_schema=CalculatorInput,
    )

    # 测试
    tests = [
        {"a": 10, "b": 3, "operation": "add"},
        {"a": 10, "b": 3, "operation": "multiply"},
        {"a": 10, "b": 0, "operation": "divide"},
    ]

    for t in tests:
        result = calculator.invoke(t)
        print(f"  {result}")
    print()


# ============================================================
# 3. bind_tools 手动测试工具调用
# ============================================================

def demo_bind_tools():
    print("=" * 50)
    print(f"Demo 9-1 (3/3): bind_tools 手动测试 [{QWEN_MODEL}]")
    print("=" * 50)

    @tool
    def search_weather(city: str) -> str:
        """查询城市天气"""
        return {"北京": "晴天 25°C", "上海": "多云 22°C"}.get(city, "未找到")

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {e}"

    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools([search_weather, calculator])

    # 测试1：需要工具
    response = llm_with_tools.invoke("北京天气怎么样？")
    print(f"问题: 北京天气怎么样？")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  LLM 选择工具: {tc['name']}({tc['args']})")
            # 手动执行
            tools_map = {"search_weather": search_weather, "calculator": calculator}
            result = tools_map[tc["name"]].invoke(tc["args"])
            print(f"  工具结果: {result}")
    print()

    # 测试2：需要计算
    response = llm_with_tools.invoke("123 * 456 等于多少？")
    print(f"问题: 123 * 456 等于多少？")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  LLM 选择工具: {tc['name']}({tc['args']})")
            tools_map = {"search_weather": search_weather, "calculator": calculator}
            result = tools_map[tc["name"]].invoke(tc["args"])
            print(f"  工具结果: {result}")
    print()

    # 测试3：不需要工具
    response = llm_with_tools.invoke("你好，介绍一下你自己")
    print(f"问题: 你好，介绍一下你自己")
    print(f"  LLM 直接回答: {response.content[:100]}")
    print(f"  工具调用: {response.tool_calls if response.tool_calls else '无'}")
    print()


if __name__ == "__main__":
    demo_basic_tool()
    demo_structured_tool()
    demo_bind_tools()

    print("=" * 50)
    print("Demo 9-1 完成!")
    print()
    print("工具定义方式:")
    print("  @tool 装饰器   - 最简单，推荐大多数场景")
    print("  StructuredTool  - 需要精细控制时使用")
    print("  bind_tools()    - 手动测试 LLM 的工具选择能力")
    print("=" * 50)
