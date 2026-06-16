"""
Step 4 · 评测 student vs teacher

内在评测（本脚本）：
  - 改写质量：LLM-as-judge（保留原意 / 自然成句 / 无幻觉形容词）+ 与 teacher 改写的 bge 语义相似度
  - filter_expr 正确性：与 teacher 的 expr 是否等价 / 能否在 Milvus 跑通

外在评测（在主项目跑，不在这里）：
  cd ../PythonProject1 && .venv/bin/python evals/run_rag_eval.py
  三方对比 baseline / DeepSeek(teacher) / Qwen(student)，看 Recall@K / NDCG / HitRate。
  注意：run_rag_eval 目前仍引用旧的分离组件，接 student 前需先对齐到 understand()。

延迟 & 成本：记录 student 本地推理延迟 vs DeepSeek API 往返 + token 成本。

运行：
    ../.venv/bin/python scripts/4_evaluate.py
"""
from __future__ import annotations

from pathlib import Path

TEST_SET = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "test.jsonl"


def main() -> None:
    # TODO（待定稿）：
    # 1. 读 test.jsonl
    # 2. student 推理（复用 3_infer）
    # 3. 改写：LLM-judge + 语义相似度；filter：等价性 / Milvus 可执行性
    # 4. 汇总打印
    raise NotImplementedError


if __name__ == "__main__":
    main()
