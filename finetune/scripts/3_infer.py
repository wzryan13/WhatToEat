"""
Step 3 · 加载 LoRA adapter 做单条推理（sanity check）

把微调后的 student 当成 teacher 的平替试一试：
    输入 query → 输出 {"rewritten_query": ..., "filter_expr": ...}

运行：
    ../.venv/bin/python scripts/3_infer.py "今晚吃啥"
"""
from __future__ import annotations

import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parents[1] / "outputs"


def infer(query: str) -> dict:
    # TODO（待定稿）：
    # 1. 加载 base model + PeftModel.from_pretrained(ADAPTER_DIR) 到 mps
    # 2. 套和训练一致的 chat template，generate
    # 3. parse 出 JSON {rewritten_query, filter_expr}，做兜底
    raise NotImplementedError


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "今晚吃啥"
    print(infer(q))
