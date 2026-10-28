# course-rag · 基于个人笔记的 RAG 问答系统

> 用 **LangChain + DeepSeek + bge-small-zh** 搭的检索增强问答系统。
> 它只根据「我自己的学习笔记」回答问题，**每句回答都带出处**，查不到就老实说不知道。

---

## 效果示例

```
❓ 线程和进程有什么区别？

📎 检索到的资料：
   1. 17-线程与进程.md  [0.705]
   2. 17-线程与进程.md  [0.697]
   ...

💬 回答：
| 对比项 | 进程（Process） | 线程（Thread） |
| --- | --- | --- |
| 是什么 | 一个独立运行的程序实例 | 进程里的执行单元 |
| 内存   | 各自独立，互不共享 | 共享同一进程内存 |
| 开销   | 创建/切换贵 | 轻量、快 |
| 通信   | 要队列/管道/共享内存 | 直接读写同一变量（但要加锁） |

出处：17-线程与进程.md

📚 用到的笔记：17-线程与进程.md
```

**查不到时不会硬编**：

```
❓ 明天天气怎么样

📎 最高相似度：0.316（低于阈值 0.45）
💬 我的笔记里没有相关内容。
```

---

## 技术栈

| 环节 | 用了什么 | 为什么 |
| --- | --- | --- |
| 大模型 | DeepSeek（`deepseek-chat`） | OpenAI 兼容，用官方 `langchain-openai` 即可，无需第三方插件 |
| 向量化 | **fastembed** + `BAAI/bge-small-zh-v1.5` | ONNX 实现，**不用装 2.5GB 的 torch**；512 维中文模型，90MB |
| 向量检索 | numpy 手写余弦相似度 | 零额外依赖，每一步都看得见（生产可换 Chroma/FAISS） |
| 编排 | LangChain LCEL | `prompt \| llm \| StrOutputParser()` 一条链 |
| 索引缓存 | `.npz`（numpy 压缩包） | 建一次用多次，避免重复向量化 |

> ⚠️ **DeepSeek 没有 embedding 接口**，所以向量化必须另找模型——这是本项目的第一个关键决策。

---

## 架构

```
建索引（只需一次）：
    notes/*.md
        │  loader.load_docs()
        ▼
    27 篇笔记
        │  splitter.split_text()  ← 200 字/段，加《标题》前缀
        ▼
    501 个片段
        │  fastembed 向量化（512 维）
        ▼
    data/index.npz   ← 缓存到磁盘

问答（每次）：
    你的问题
        │ 向量化
        ▼
    余弦相似度 × 501 段
        │ 排序取 top-6
        ▼
    相似度 < 0.45 ？
        ├── 是 → 直接回「笔记里没有相关内容」（防幻觉）
        └── 否 → 拼进提示词 → DeepSeek 生成 → 带出处返回
```

---

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配置
cp .env.example .env        # Windows: copy .env.example .env
# 在 .env 里填入 DEEPSEEK_API_KEY 和 NOTES_DIR（你的笔记目录）

# 3. 建索引（第一次会下载 90MB 模型）
python build_index.py

# 4. 提问
python ask.py                          # 交互模式
python ask.py "线程和进程的区别"        # 单次提问
```

---

## 关键设计（值得说明的四个决定）

### 1. 阈值拒答：宁愿说不知道，也不要幻觉

```python
if hits[0].score < self.min_score:      # min_score = 0.45
    return Answer(text="我的笔记里没有相关内容。", refused=True)
```

实测：相关问题的最高分在 **0.65~0.72**，无关问题在 **0.28~0.36**，阈值 0.45 能干净地分开两类。

### 2. top-k 取 6，不是 1

排第一的片段**不一定**是真正答案。实测「怎么用 dataclass 让对象不可变」时，第 1 名是讲"类属性"的笔记，真正的 `frozen=True` 内容排在第 4~6 名。取 6 条一起喂给模型才稳。

### 3. 给每个片段加《标题》前缀（提升命中率的关键）

```python
self.chunks.append(Chunk(text=f"《{title}》\n{piece}", source=doc.name))
```

**加之前**：讲 `frozen` 的片段排在第 24 / 39 / 96 名，检索完全命中不了。
**加之后**：直接进 top-3。

原因：片段自带主题标签后，向量不再只由段落内容决定，主题归属更清晰。

### 4. 切分 200 字 + 40 重叠

一开始用 400 字，结果 `frozen` 的解释被和别的段落混在一大块里，向量被"平均"掉。改成 200 字后每块主题更聚焦。

> 这里踩过的完整排查过程见下方「踩坑记录」。

---

## 踩坑记录

| 坑 | 现象 | 解法 |
| --- | --- | --- |
| DeepSeek 不支持 `json_schema` | 结构化输出报 `This response_format type is unavailable now` | `with_structured_output(..., method="function_calling")` |
| DeepSeek 没有 embedding | 无法向量化 | 用本地 `bge-small-zh-v1.5`（fastembed ONNX 版） |
| 检索命中不了细节 | 问 `frozen` 却答"没相关内容" | 切分 400→200 + 片段加《标题》前缀 + top_k 3→6 |
| 装 `sentence-transformers` 太重 | 要装 2.5GB torch | 换 `fastembed`（ONNX），只需 ~100MB |
| HuggingFace 下载慢 | 模型下不动 | 设 `HF_ENDPOINT=https://hf-mirror.com` |

---

## 目录结构

```
course-rag/
├── config.py            # 读 .env、创建 DeepSeek 模型
├── build_index.py       # 建索引（读 → 切 → 向量化 → 存盘）
├── ask.py               # 提问（交互 or 单次）
├── rag/
│   ├── loader.py        # 读 Markdown，去掉 YAML 元信息
│   ├── splitter.py      # 按段落切分 + 重叠
│   ├── store.py         # 向量库（建库 / 存取 / 余弦检索）
│   └── qa.py            # RAG 问答（阈值判断 + 拼提示词 + 出处）
├── data/index.npz       # 索引缓存（gitignore，可重建）
└── requirements.txt
```

---

## 未来改进

- [ ] 换成 Chroma / FAISS，支持增量添加（现在每加一篇笔记要重建全量索引）
- [ ] 混合检索：向量 + BM25 关键词，提升 top-1 命中率
- [ ] Rerank 模型（`bge-reranker`）对 top-20 重排
- [ ] 多轮对话：记住上下文再追问（接 LangGraph 的 checkpointer）
- [ ] 流式输出（`llm.stream`）
- [ ] Web 界面 / FastAPI 接口
- [ ] 评估集：手工标注 30 个问题，量化命中率

---

## 说明

本项目是个人学习项目，用于练习 RAG 全流程：**切分策略、embedding 选型、检索调优、防幻觉**。
代码优先保证"看得懂、跑得起来"，而不是生产级性能。
