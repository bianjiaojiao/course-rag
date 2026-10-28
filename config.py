"""course-rag 的配置：读取 .env，创建 DeepSeek 模型对象。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# 知识库目录（笔记所在位置）
NOTES_DIR = Path(
    os.getenv("NOTES_DIR", r"C:/Users/边角/Desktop/py-bootcamp/notes/topics")
)

# 索引缓存文件
INDEX_FILE = ROOT / "data" / "index.npz"


def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOpenAI:
    """创建 DeepSeek 模型对象。Key 缺失时给出中文提示。"""
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "未找到 DEEPSEEK_API_KEY。\n"
            "请把 .env.example 复制成 .env，并填入你的 Key。"
        )
    return ChatOpenAI(
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
    )
