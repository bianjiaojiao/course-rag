"""读文档：把目录下的 Markdown 笔记读成 Doc 列表。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Doc:
    """一篇笔记。"""

    name: str  # 文件名，如 "17-线程与进程.md"
    text: str  # 正文（已去掉开头的 YAML 元信息）


def load_docs(directory: Path) -> list[Doc]:
    """读入目录下所有 .md 文件。"""
    if not directory.exists():
        raise SystemExit(f"知识库目录不存在：{directory}")

    docs: list[Doc] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        # 去掉开头的 --- ... --- 元信息块
        body = re.sub(r"^---\n.*?\n---\n", "", raw, count=1, flags=re.S)
        docs.append(Doc(name=path.name, text=body))
    return docs
