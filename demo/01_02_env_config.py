"""
第1章 Demo 2：.env 配置生成与加载演示

演示如何创建 .env 文件、加载环境变量、安全管理 API Key。
可独立运行。
"""

import os
from pathlib import Path


def create_env_file():
    """演示创建 .env 文件"""
    # 定位到项目根目录
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    gitignore_path = project_root / ".gitignore"

    print("=" * 50)
    print("Demo 2: .env 配置管理")
    print("=" * 50)
    print()

    # 检查 .env 是否已存在
    if env_path.exists():
        print(f".env 文件已存在: {env_path}")
        print("内容预览:")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # 脱敏显示
                    key = line.split("=")[0]
                    print(f"  {key}=***")
                elif line:
                    print(f"  {line}")
    else:
        print(f".env 文件不存在，正在创建模板: {env_path}")
        template = """\
# ===== GLM (智谱AI) API 配置 =====
# API 兼容 OpenAI 格式，通过 base_url 指向智谱 AI
GLM_API_KEY=your-glm-api-key-here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4-flash
"""
        with open(env_path, "w") as f:
            f.write(template)
        print(f"✓ 已创建 .env 模板文件: {env_path}")
        print("  请编辑该文件，填入你的真实 API Key")

    print()

    # 检查 .gitignore
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            content = f.read()
        if ".env" in content:
            print("✓ .gitignore 已包含 .env（API Key 不会被提交）")
        else:
            print("⚠ .gitignore 未包含 .env，建议添加")
    else:
        print("⚠ 未找到 .gitignore，建议创建并添加 .env")

    print()
    return env_path


def demo_load_env(env_path: Path):
    """演示加载和使用环境变量"""
    print("--- 加载 .env ---")

    try:
        from dotenv import load_dotenv

        loaded = load_dotenv(env_path)
        print(f"load_dotenv 返回: {loaded} (True 表示文件已加载)")

        # 检查 GLM 配置
        config_keys = ["GLM_API_KEY", "GLM_BASE_URL", "GLM_MODEL"]
        for key in config_keys:
            value = os.environ.get(key, "")
            if value and not value.startswith("your-"):
                if "KEY" in key:
                    masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    print(f"  {key}: {masked}")
                else:
                    print(f"  {key}: {value}")
            elif value:
                print(f"  {key}: 仍为占位符")
            else:
                print(f"  {key}: 未设置")
    except ImportError:
        print("python-dotenv 未安装，请运行: uv add python-dotenv")

    print()

    # 安全提醒
    print("--- 环境变量安全提醒 ---")
    print("✓ 推荐: 使用 .env 文件 + python-dotenv")
    print("✗ 不推荐: 在代码中硬编码 API Key")
    print("✗ 不推荐: 将 .env 文件提交到 Git")
    print()

    # 验证 key
    key = os.environ.get("GLM_API_KEY", "")
    base_url = os.environ.get("GLM_BASE_URL", "")
    if key and not key.startswith("your-") and base_url:
        print("✓ GLM API 配置看起来正确")
    elif key:
        print("⚠ GLM_API_KEY 仍为占位符，请替换为真实 Key")
    else:
        print("⚠ GLM_API_KEY 未设置")


def demo_glm_config():
    """演示 GLM 连接配置"""
    print()
    print("--- GLM (智谱AI) 配置说明 ---")

    api_key = os.environ.get("GLM_API_KEY", "")
    base_url = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    model = os.environ.get("GLM_MODEL", "glm-4-flash")

    print(f"  API 地址: {base_url}")
    print(f"  模型:     {model}")
    print()

    if api_key and not api_key.startswith("your-"):
        print("✓ GLM API Key 已配置，可以调用 LLM")
        print()
        print("在代码中使用 GLM:")
        print(f"""
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="{model}",
      api_key=os.environ["GLM_API_KEY"],
      base_url=os.environ["GLM_BASE_URL"],
  )
  response = llm.invoke("你好")
  print(response.content)
""")
    else:
        print("⚠ 请在 .env 中设置 GLM_API_KEY")
        print("  获取地址: https://open.bigmodel.cn/")


if __name__ == "__main__":
    env_path = create_env_file()
    demo_load_env(env_path)
    demo_glm_config()
