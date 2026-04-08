"""
第1章 Demo 1：环境验证脚本

验证 Python 版本、已安装的 LangChain 相关包、API Key 配置是否正确。
可独立运行，帮助快速排查环境问题。
"""

import sys


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    print(f"[1] Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 10:
        print("    ✓ Python 版本满足要求 (>= 3.10)")
    else:
        print("    ✗ Python 版本过低，建议升级到 3.10+")
    print()


def check_packages():
    """检查关键包是否已安装"""
    packages = {
        "langchain": "LangChain 核心",
        "langchain_openai": "LangChain OpenAI 集成",
        "langchain_core": "LangChain 核心 (langchain-core)",
        "langgraph": "LangGraph",
        "dotenv": "python-dotenv",
    }

    print("[2] 包安装检查:")
    for pkg, desc in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "已安装")
            print(f"    ✓ {desc} ({pkg}): {version}")
        except ImportError:
            print(f"    ✗ {desc} ({pkg}): 未安装")
    print()


def check_api_keys():
    """检查 API Key 是否已配置"""
    import os
    from pathlib import Path

    print("[3] API Key 检查:")

    # 尝试加载 .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"    .env 文件: {env_path}")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print("    ✓ .env 加载成功")
        except Exception as e:
            print(f"    ✗ .env 加载失败: {e}")
    else:
        print(f"    ! 未找到 .env 文件 (预期路径: {env_path})")
        print("      请参考 docs/01-environment-setup.md 创建 .env 文件")

    # 检查 GLM API Key
    keys = {
        "GLM_API_KEY": "智谱AI (GLM)",
        "GLM_BASE_URL": "GLM API 地址",
        "GLM_MODEL": "GLM 模型名称",
    }
    for key, provider in keys.items():
        value = os.environ.get(key, "")
        if value and not value.startswith("your-"):
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"    ✓ {provider} ({key}): {masked}")
        elif value:
            print(f"    ✗ {provider} ({key}): 仍为占位符")
        else:
            print(f"    ✗ {provider} ({key}): 未设置")
    print()


def quick_llm_test():
    """快速 LLM 调用测试"""
    import os

    api_key = os.environ.get("GLM_API_KEY", "")
    base_url = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    model = os.environ.get("GLM_MODEL", "glm-4-flash")

    if not api_key or api_key.startswith("your-"):
        print("[4] LLM 调用测试: 跳过 (GLM_API_KEY 未设置或仍为占位符)")
        return

    print(f"[4] LLM 调用测试 (模型: {model}):")
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=50,
            api_key=api_key,
            base_url=base_url,
        )
        response = llm.invoke("请用一句话回答：1+1等于几？")
        print(f"    ✓ 调用成功")
        print(f"    回复: {response.content}")

        # Token 用量
        usage = response.response_metadata.get("token_usage", {})
        if usage:
            print(f"    Token 用量: 输入 {usage.get('prompt_tokens')} + 输出 {usage.get('completion_tokens')} = 总计 {usage.get('total_tokens')}")
    except Exception as e:
        print(f"    ✗ 调用失败: {e}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("LangChain 环境验证")
    print("=" * 50)
    print()

    check_python_version()
    check_packages()
    check_api_keys()
    quick_llm_test()

    print("=" * 50)
    print("验证完成。如果所有项都显示 ✓，说明环境配置正确。")
    print("=" * 50)
