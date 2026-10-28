"""建索引：读取笔记 → 切分 → 向量化 → 存到 data/index.npz

用法：
    python build_index.py            # 用默认参数建库
    python build_index.py --size 300 # 换个切分长度

建完索引后，用 `python ask.py` 提问就不用再等向量化了。
"""
from __future__ import annotations

import argparse
import time

from config import INDEX_FILE, NOTES_DIR
from rag.loader import load_docs
from rag.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="为 course-rag 建索引")
    parser.add_argument("--size", type=int, default=200, help="每段的字数上限")
    parser.add_argument("--overlap", type=int, default=40, help="硬切时的重叠字数")
    args = parser.parse_args()

    print("=" * 60)
    print("① 读笔记")
    print("=" * 60)
    docs = load_docs(NOTES_DIR)
    print(f"   目录：{NOTES_DIR}")
    print(f"   读到 {len(docs)} 篇笔记")

    print()
    print("=" * 60)
    print("② 切分 + 向量化（第一次要下载模型，约 90MB）")
    print("=" * 60)
    start = time.time()
    store = VectorStore()
    store.build(docs, chunk_size=args.size, overlap=args.overlap)
    cost = time.time() - start

    vectors = store.vectors
    assert vectors is not None
    print(f"   片段数：{len(store.chunks)}")
    print(f"   向量维度：{vectors.shape[1]}")
    print(f"   耗时：{cost:.1f} 秒")

    print()
    print("=" * 60)
    print("③ 保存索引")
    print("=" * 60)
    store.save(INDEX_FILE)
    size_mb = INDEX_FILE.stat().st_size / 1024 / 1024
    print(f"   已保存：{INDEX_FILE}（{size_mb:.1f} MB）")
    print()
    print("下一步： python ask.py")


if __name__ == "__main__":
    main()
