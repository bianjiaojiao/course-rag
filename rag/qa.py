"""RAG 问答：检索 → 拼提示词 → 调模型 → 带出处返回。"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .store import Hit, VectorStore

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是个人学习助手，只能根据【资料】回答问题。\n"
            "规则：\n"
            "1. 资料里有的，用资料内容回答，条理清晰；\n"
            "2. 资料里没有的，直接说「你的笔记里没有相关内容」，绝不编造；\n"
            "3. 最后单独一行写：出处：文件名",
        ),
        ("human", "【资料】\n{context}\n\n【问题】{question}"),
    ]
)


@dataclass
class Answer:
    """一次问答的完整结果。"""

    text: str  # 回答正文（或拒答提示）
    hits: list[Hit]  # 检索到的片段
    refused: bool  # 是否因为"查不到"而拒答

    @property
    def best_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def sources(self) -> list[str]:
        """去重后的来源文件列表。"""
        seen: list[str] = []
        for h in self.hits:
            if h.source not in seen:
                seen.append(h.source)
        return seen


class RAG:
    """检索增强问答器。"""

    def __init__(
        self,
        store: VectorStore,
        llm: ChatOpenAI,
        top_k: int = 6,
        min_score: float = 0.45,
    ) -> None:
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.chain = PROMPT | llm | StrOutputParser()

    def _build_context(self, hits: list[Hit]) -> str:
        parts = [
            f"（来自《{h.source}》，相似度 {h.score:.2f}）\n{h.text}" for h in hits
        ]
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str) -> Answer:
        hits = self.store.search(question, k=self.top_k)

        # 阈值判断：检索不到就直接说不知道，避免幻觉
        if not hits or hits[0].score < self.min_score:
            return Answer(
                text="我的笔记里没有相关内容。",
                hits=hits,
                refused=True,
            )

        text = self.chain.invoke(
            {"context": self._build_context(hits), "question": question}
        )
        return Answer(text=text, hits=hits, refused=False)
