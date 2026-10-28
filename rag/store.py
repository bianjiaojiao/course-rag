"""向量库：把片段变成向量、存盘、按相似度检索。

这一版用「numpy + 余弦相似度」手写实现，好处是：
  · 零额外依赖（不用装 Chroma / FAISS）
  · 每一步都看得见，适合学习
生产环境通常换成 Chroma / FAISS / Qdrant（它们会做索引加速）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from .loader import Doc
from .splitter import split_text

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


@dataclass
class Chunk:
    """一个片段。"""

    text: str
    source: str  # 来自哪篇笔记


@dataclass
class Hit:
    """一条检索结果。"""

    text: str
    source: str
    score: float


class VectorStore:
    """极简向量库。"""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = TextEmbedding(model_name)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    # ---------- 建库 ----------
    def build(self, docs: list[Doc], chunk_size: int = 200, overlap: int = 40) -> None:
        self.chunks = []
        for doc in docs:
            title = doc.name.removesuffix(".md")
            for piece in split_text(doc.text, chunk_size, overlap):
                # 关键技巧：给每个片段加上"来自哪篇笔记"的标题前缀
                # 这样片段自带主题标签，检索时更容易命中对应主题
                self.chunks.append(Chunk(text=f"《{title}》\n{piece}", source=doc.name))

        texts = [c.text for c in self.chunks]
        self.vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)

    # ---------- 存取 ----------
    def save(self, path: Path) -> None:
        if self.vectors is None:
            raise RuntimeError("还没建库，无法保存")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            texts=np.array([c.text for c in self.chunks], dtype=object),
            sources=np.array([c.source for c in self.chunks], dtype=object),
        )

    @classmethod
    def load(cls, path: Path, model_name: str = DEFAULT_MODEL) -> "VectorStore":
        if not path.exists():
            raise FileNotFoundError(f"索引不存在：{path}（请先运行 build_index.py）")
        store = cls(model_name)
        data = np.load(path, allow_pickle=True)
        store.vectors = data["vectors"]
        store.chunks = [
            Chunk(text=str(t), source=str(s))
            for t, s in zip(data["texts"], data["sources"])
        ]
        return store

    # ---------- 检索 ----------
    def search(self, query: str, k: int = 6) -> list[Hit]:
        if self.vectors is None or not self.chunks:
            raise RuntimeError("向量库为空")

        q = np.array(next(iter(self.model.embed([query]))), dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q)
        scores = (self.vectors @ q) / (norms + 1e-8)

        k = min(k, len(self.chunks))
        top = np.argsort(scores)[::-1][:k]
        return [
            Hit(text=self.chunks[i].text, source=self.chunks[i].source, score=float(scores[i]))
            for i in top
        ]
