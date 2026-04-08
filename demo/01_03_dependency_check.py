"""
第1章 Demo 3：依赖包检查与安装指导

检查所有章节所需的依赖包是否已安装，给出安装建议。
可独立运行，不需要 API Key。
"""

import importlib
import sys


# 各章节需要的依赖包
CHAPTER_DEPS = {
    "第1-2章 (基础)": [
        ("langchain", "langchain"),
        ("langchain_core", "langchain-core"),
        ("langchain_openai", "langchain-openai"),
        ("dotenv", "python-dotenv"),
    ],
    "第3-4章 (Prompt与解析)": [
        ("langchain_core", "langchain-core"),
        ("pydantic", "pydantic"),
    ],
    "第5章 (Chain)": [
        ("langchain", "langchain"),
    ],
    "第6-7章 (RAG检索)": [
        ("langchain_community", "langchain-community"),
        ("chromadb", "chromadb"),
        ("pypdf", "pypdf"),
        ("bs4", "beautifulsoup4"),
    ],
    "第8章 (Memory)": [
        ("langchain_community", "langchain-community"),
    ],
    "第9-12章 (Tools与Agent)": [
        ("langgraph", "langgraph"),
        ("httpx", "httpx"),
    ],
    "第13-15章 (LangGraph进阶)": [
        ("langgraph", "langgraph"),
    ],
}


def check_package(import_name: str) -> tuple[bool, str]:
    """检查单个包是否可导入，返回 (是否已安装, 版本号)"""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "未知版本")
        return True, version
    except ImportError:
        return False, ""


def main():
    print("=" * 60)
    print("LangChain 学习路线 - 依赖包检查")
    print("=" * 60)
    print()

    missing_packages = []
    install_commands = []

    for chapter, deps in CHAPTER_DEPS.items():
        print(f"📚 {chapter}")
        all_ok = True

        for import_name, pip_name in deps:
            installed, version = check_package(import_name)
            if installed:
                print(f"   ✓ {pip_name:30s} {version}")
            else:
                print(f"   ✗ {pip_name:30s} 未安装")
                all_ok = False
                if pip_name not in missing_packages:
                    missing_packages.append(pip_name)

        if all_ok:
            print(f"   → 状态: 就绪 ✓")
        else:
            print(f"   → 状态: 需要安装缺失的包")
        print()

    # 安装建议
    if missing_packages:
        print("=" * 60)
        print("安装命令:")
        print()
        print("使用 uv (推荐):")
        uv_cmd = "uv add " + " ".join(missing_packages)
        print(f"  {uv_cmd}")
        print()
        print("使用 pip:")
        pip_cmd = "pip install " + " ".join(missing_packages)
        print(f"  {pip_cmd}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✓ 所有依赖包已安装！你可以开始学习了。")
        print("=" * 60)

    print()
    print("提示: 各章节的 demo 程序位于 demo/ 目录中")
    print("运行方式: python demo/01_01_verify_setup.py")


if __name__ == "__main__":
    main()
