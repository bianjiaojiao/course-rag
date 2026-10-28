"""切分：把长文切成小段（chunk）。

为什么要切分？
  1. 模型一次能看的字数有限（上下文窗口）
  2. 检索需要"精细命中"——整篇塞进去，向量会被稀释成平均值
"""
from __future__ import annotations


def split_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """按空行分段，再把段落合并成不超过 chunk_size 的块。

    Args:
        text: 原文
        chunk_size: 每块最大字数
        overlap: 超长段落硬切时的重叠字数（避免一句话被切断）

    Returns:
        切好的片段列表
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= chunk_size:
            # 还能装下，继续攒
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            if len(para) > chunk_size:
                # 单段本身就超长 → 带重叠硬切
                step = max(1, chunk_size - overlap)
                for i in range(0, len(para), step):
                    chunks.append(para[i : i + chunk_size])
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append(buffer)
    return chunks
