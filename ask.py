"""提问：用你的笔记回答问题，并给出处。

用法：
    python ask.py                      # 交互模式（可以连续问）
    python ask.py "线程和进程的区别"    # 只问一个

首次使用请先跑： python build_index.py
"""
from __future__ import annotations

import sys

from config import INDEX_FILE, get_llm
from rag.qa import RAG
from rag.store import VectorStore


def print_answer(rag: RAG, question: str) -> None:
    result = rag.ask(question)

    print()
    print("=" * 64)
    print(f"❓ {question}")
    print("=" * 64)

    if result.refused:
        print(f"📎 最高相似度：{result.best_score:.3f}（低于阈值 {rag.min_score}）")
        print(f"💬 {result.text}")
        return

    print("📎 检索到的资料：")
    for rank, hit in enumerate(result.hits, 1):
        print(f"   {rank}. {hit.source}  [{hit.score:.3f}]")

    print()
    print("💬 回答：")
    print(result.text)

    print()
    print(f"📚 用到的笔记：{'、'.join(result.sources)}")


def main() -> None:
    store = VectorStore.load(INDEX_FILE)
    rag = RAG(store, get_llm(temperature=0))

    # 命令行直接带问题
    if len(sys.argv) > 1:
        print_answer(rag, " ".join(sys.argv[1:]))
        return

    # 交互模式
    print("=" * 64)
    print("course-rag · 基于你的笔记回答问题")
    print(f"知识库：{len(store.chunks)} 个片段")
    print("输入问题回车即可，输入 q 退出")
    print("=" * 64)

    while True:
        try:
            question = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            print("再见 👋")
            break

        print_answer(rag, question)


if __name__ == "__main__":
    main()
