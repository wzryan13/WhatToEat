"""
Step 1 · 构建 SFT 数据集（方案 B：查询理解蒸馏）

流程：
    datasets/raw/ 的原始 query
      → teacher: QueryUnderstandingModule.understand(query, CATALOG)
      → (rewritten_query, filter_expr)
      → 清洗（去重 / 丢弃 teacher 翻车样本）
      → datasets/processed/{train,val,test}.jsonl
         Alpaca 三段式，output = 结构化 JSON 字符串：
           {"rewritten_query": "...", "filter_expr": "..." | null}

运行：
    ../.venv/bin/python scripts/1_build_dataset.py
    需要 DEEPSEEK_API_KEY（从主项目软链 .env）。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 让脚本能 import 主项目代码（本 worktree 根含完整 WhatToEat 源码）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = Path(__file__).resolve().parents[1] / "datasets" / "raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "processed"


# TODO（待与 Ryan 定稿）：student 的指令模板。
#   - 要不要把完整 metadata_catalog 写进 instruction？（filter_expr 依赖它）
#   - 与生产 query_understanding 的 prompt 对齐到什么程度？
INSTRUCTION = "TODO: 查询理解指令模板（含可用 category/difficulty 取值说明）"

# TODO：metadata_catalog 来源。主项目 evals/run_rag_eval.py 里有 METADATA_CATALOG 常量，
#   可直接复用，或从 rag_service 动态取。
CATALOG: dict = {}


def load_raw_queries() -> list[str]:
    """读取 datasets/raw/ 下的原始 query（每行一条 / jsonl，待定格式）。"""
    # TODO: 等 Ryan 给 query 后定文件格式
    raise NotImplementedError


async def label_with_teacher(queries: list[str]) -> list[dict]:
    """调用主项目 teacher 给每条 query 打 (rewritten_query, filter_expr) 标签。"""
    from rag.pipeline.query_understanding import QueryUnderstandingModule

    teacher = QueryUnderstandingModule()
    samples: list[dict] = []
    for q in queries:
        rewritten, filter_expr = await teacher.understand(q, CATALOG)
        samples.append({
            "instruction": INSTRUCTION,
            "input": q,
            "output": json.dumps(
                {"rewritten_query": rewritten, "filter_expr": filter_expr},
                ensure_ascii=False,
            ),
        })
    return samples


def clean(samples: list[dict]) -> list[dict]:
    """去重 + 丢弃 teacher 翻车样本（幻觉形容词 / 关键词堆砌 / 未改写等）。"""
    # TODO: 定清洗规则（人工扫一批后确定）
    return samples


def split_and_dump(samples: list[dict], ratio=(0.8, 0.1, 0.1)) -> None:
    """切 train/val/test 并写 jsonl。"""
    # TODO
    raise NotImplementedError


async def main() -> None:
    queries = load_raw_queries()
    samples = await label_with_teacher(queries)
    samples = clean(samples)
    split_and_dump(samples)


if __name__ == "__main__":
    asyncio.run(main())
