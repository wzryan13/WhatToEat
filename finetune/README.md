# finetune — RAG 查询理解蒸馏（方案 B）

把 WhatToEat 主项目里 RAG「查询理解」节点从 **DeepSeek（云端）** 蒸馏到本地
**Qwen2.5-3B-Instruct + LoRA**，目标是**生产替换 + 作为实习项目展示**。

## 这个子项目在做什么

主项目当前用 [`rag/pipeline/query_understanding.py`](../rag/pipeline/query_understanding.py)
的 `QueryUnderstandingModule.understand(query, metadata_catalog)` 做一次 DeepSeek 调用，
输出 `(rewritten_query, filter_expr)`。

我们用它当 **teacher** 蒸馏一个本地小模型 **student**，让 student 学会同样的
`query (+catalog) → {rewritten_query, filter_expr}`，最终替换掉那次云端调用。

> **方案 B**：student 同时学「查询改写」+「元数据过滤表达式」两个输出（完整替换，降本完整）。

## 与主项目的关系

- 本目录是 git **worktree**（分支 `feat/finetune-query-understanding`），主分支 `main` 在
  `../PythonProject1`，两边并行、互不污染。
- **teacher 复用**主项目代码（本 worktree 内含完整 WhatToEat 源码）。
- **外在评测复用**主项目 `evals/run_rag_eval.py`（Recall@K / NDCG / HitRate）。
- 训练产物（LoRA adapter）最终接回主项目 `query_understanding`。

## 目录结构

```
finetune/
├── README.md
├── requirements.txt          # 独立 venv 的依赖
├── .gitignore                # 排除权重/checkpoint/生成数据
├── configs/
│   └── lora.yaml             # LoRA 超参 + 训练配置
├── datasets/
│   ├── raw/                  # ← 你手挑的原始 query 种子（进 git）
│   └── processed/            # teacher 打标签后的 train/val/test（gitignore）
├── scripts/
│   ├── 1_build_dataset.py    # query → teacher 打标签 → SFT jsonl
│   ├── 2_train_lora.py       # Qwen2.5-3B-Instruct + LoRA 训练
│   ├── 3_infer.py            # 加载 adapter 单条推理（sanity check）
│   └── 4_evaluate.py         # 内在评测（+ 指引外在评测）
└── outputs/                  # LoRA adapter / checkpoint（gitignore）
```

## 环境准备

```bash
# 1. 装依赖（独立 venv，已建在 ../.venv）
../.venv/bin/pip install -r requirements.txt

# 2. teacher 要调 DeepSeek，但 worktree 不含 .env（被 gitignore）
#    从主项目软链一份过来：
ln -s ../../PythonProject1/.env ../.env
```

> ⚠️ Python 3.13 / Apple M4 MPS / 无 CUDA → **不要装 bitsandbytes**（QLoRA 在 Mac 不可用）。

## 流程（对应 scripts/1-4）

| 步骤 | 脚本 | 输入 → 输出 |
|---|---|---|
| 1. 造数据 | `1_build_dataset.py` | `datasets/raw/*` → `datasets/processed/{train,val,test}.jsonl` |
| 2. 训练 | `2_train_lora.py` | processed/train.jsonl → `outputs/`（LoRA adapter） |
| 3. 试推理 | `3_infer.py` | 单条 query → `{rewritten_query, filter_expr}` |
| 4. 评测 | `4_evaluate.py` | 内在指标；外在用主项目 `evals/run_rag_eval.py` |

## 状态

🚧 骨架已搭，脚本为占位（含 TODO）。下一步：拿到 query 种子后定稿 `1_build_dataset.py`。
