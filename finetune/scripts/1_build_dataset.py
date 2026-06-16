"""
Step 1 · 构建 SFT 数据集（方案 B：查询理解蒸馏 / 自给标签 / 完整 prompt）

输入：datasets/raw/ 下的已标注源文件（见 RAW_FILES），每条 {id, query, rewritten_query, expr}
      文件可为数组，或质检输出的 {"fixed": [...]} 格式。
输出：datasets/processed/
    train.jsonl   训练集（Alpaca 三段）
    test.jsonl    held-out 测试集（原始标注格式，训练绝不碰，评估时再渲染）
    smoke.jsonl   从 train 抽 SMOKE_N 条冒烟子集

instruction 复用生产 QUERY_UNDERSTANDING_PROMPT，并补一句严格 JSON 指令：
    student 是裸 generate，没有生产 with_structured_output 的 API 层 schema 约束，
    必须在 prompt 文本里显式要求 JSON，否则学不会输出结构化结果。

运行：
    .venv/bin/python finetune/scripts/1_build_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = FT_DIR / "datasets" / "raw"
OUT_DIR = FT_DIR / "datasets" / "processed"

# 要合并的已标注源文件（每个可为数组，或质检输出的 {"fixed": [...]} 格式）
RAW_FILES = [
    "finetune_queries_v1_labeled.json",
    "批量微调数据.json",
]

SMOKE_N = 30
TEST_RATIO = 0.1
BASE_MODEL = "/Users/wenzhouzhou/Qwen2.5-3B-Instruct"

CATALOG = {
    "recipe_chunks": {
        "category": ["早餐", "汤类", "主食", "甜品", "饮品", "调料",
                     "半成品加工", "水产", "荤菜", "素菜"],
        "difficulty": ["入门", "简单", "中等", "较难", "困难"],
    }
}

# student 没有生产的 structured_output 约束，必须在 prompt 末尾显式要求 JSON
JSON_DIRECTIVE = (
    "\n\n【输出格式·严格】只输出一个 JSON 对象，形如 "
    '{"rewritten_query": "改写后的完整句子", "filter_expr": "<Milvus 过滤表达式>"}，'
    "无法可靠生成过滤时 filter_expr 取 null。不要输出任何解释、复述或多余文字。"
)


def build_instruction_renderer():
    """复用生产 prompt（importlib 绕过 rag 包依赖），末尾补 JSON 指令。"""
    import importlib.util

    qu_path = PROJECT_ROOT / "rag" / "pipeline" / "query_understanding.py"
    spec = importlib.util.spec_from_file_location("qu_standalone", qu_path)
    qu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qu)

    reference = qu._load_reference_material()
    metadata_schema = qu._summarize_metadata(CATALOG)

    def render(query: str) -> str:
        msgs = qu.QUERY_UNDERSTANDING_PROMPT.format_messages(
            query=query,
            reference_material=reference,
            metadata_schema=metadata_schema,
        )
        return msgs[0].content + JSON_DIRECTIVE

    return render


def to_sample(row: dict, render) -> dict:
    expr = (row.get("expr") or "").strip()
    filter_expr = None if expr.upper() == "NONE" or expr == "" else expr
    output = json.dumps(
        {"rewritten_query": row["rewritten_query"], "filter_expr": filter_expr},
        ensure_ascii=False,
    )
    return {"instruction": render(row["query"]), "input": "", "output": output}


def dump_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for name in RAW_FILES:
        data = json.loads((RAW_DIR / name).read_text(encoding="utf-8"))
        rows.extend(data["fixed"] if isinstance(data, dict) and "fixed" in data else data)
    seen: set[str] = set()
    return [r for r in rows if not (r["query"] in seen or seen.add(r["query"]))]


def report_token_lengths(samples: list[dict]) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    lens = []
    for s in samples:
        user = f"<|im_start|>user\n{s['instruction']}{s['input']}<|im_end|>\n<|im_start|>assistant\n"
        full = user + s["output"] + "<|im_end|>"
        lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
    lens.sort()
    mx, p95 = lens[-1], lens[int(len(lens) * 0.95)]
    suggested = ((mx // 256) + 1) * 256
    print(f"  序列 token: min={lens[0]} p95={p95} max={mx} → 建议 max_seq_length ≥ {suggested}")


def main() -> None:
    rows = load_rows()

    # held-out test split（固定 seed，训练集绝不包含 test）
    random.seed(42)
    random.shuffle(rows)
    n_test = max(1, int(len(rows) * TEST_RATIO))
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    render = build_instruction_renderer()
    train = [to_sample(r, render) for r in train_rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_jsonl(OUT_DIR / "train.jsonl", train)
    dump_jsonl(OUT_DIR / "smoke.jsonl", train[:SMOKE_N])
    dump_jsonl(OUT_DIR / "test.jsonl", test_rows)  # 原始标注，评估时再渲染

    none_n = sum(1 for r in train_rows if (r.get("expr") or "").strip().upper() == "NONE")
    print(f"✓ train.jsonl: {len(train)} 条")
    print(f"✓ test.jsonl : {len(test_rows)} 条（held-out，训练不碰）")
    print(f"✓ smoke.jsonl: {min(SMOKE_N, len(train))} 条")
    print(f"  train 标签分布: filter_expr=null {none_n} / 有表达式 {len(train_rows) - none_n}")
    report_token_lengths(train)


if __name__ == "__main__":
    main()
