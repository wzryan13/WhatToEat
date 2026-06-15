"""
Step 3 · 加载 LoRA adapter，在 held-out test 集上评估

读 datasets/processed/test.jsonl（训练绝对没见过的数据），对每条 query 渲染（含 JSON 指令，
与训练完全一致）greedy 解码，统计：合法 JSON 率 + filter_expr 命中率。

运行：
    .venv/bin/python finetune/scripts/3_infer.py                 # test 集前 N_EVAL 条
    .venv/bin/python finetune/scripts/3_infer.py "今晚吃啥"        # 单条 ad-hoc
    .venv/bin/python finetune/scripts/3_infer.py --adapter smoke  # 指定 adapter 子目录
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL = "/Users/wenzhouzhou/Qwen2.5-3B-Instruct"
OUTPUTS = FT_DIR / "outputs"
TEST_FILE = FT_DIR / "datasets" / "processed" / "test.jsonl"
N_EVAL = 20

CATALOG = {
    "recipe_chunks": {
        "category": ["早餐", "汤类", "主食", "甜品", "饮品", "调料",
                     "半成品加工", "水产", "荤菜", "素菜"],
        "difficulty": ["入门", "简单", "中等", "较难", "困难"],
    }
}

JSON_DIRECTIVE = (
    "\n\n【输出格式·严格】只输出一个 JSON 对象，形如 "
    '{"rewritten_query": "改写后的完整句子", "filter_expr": "<Milvus 过滤表达式>"}，'
    "无法可靠生成过滤时 filter_expr 取 null。不要输出任何解释、复述或多余文字。"
)


def build_renderer():
    import importlib.util

    p = PROJECT_ROOT / "rag" / "pipeline" / "query_understanding.py"
    spec = importlib.util.spec_from_file_location("qu_standalone", p)
    qu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qu)
    reference = qu._load_reference_material()
    schema = qu._summarize_metadata(CATALOG)
    return lambda q: qu.QUERY_UNDERSTANDING_PROMPT.format_messages(
        query=q, reference_material=reference, metadata_schema=schema
    )[0].content + JSON_DIRECTIVE


def norm_expr(e) -> str:
    return "NONE" if e is None else str(e).strip().upper()


def main() -> None:
    argv = sys.argv[1:]
    adapter_name = "qu-lora-full"
    if "--adapter" in argv:
        i = argv.index("--adapter")
        adapter_name = f"qu-lora-{argv[i + 1]}"
        del argv[i:i + 2]

    if argv:
        cases = [{"query": q} for q in argv]
    else:
        cases = [json.loads(l) for l in TEST_FILE.read_text(encoding="utf-8").splitlines()][:N_EVAL]

    adapter_dir = OUTPUTS / adapter_name
    print(f"加载 base + adapter（{adapter_dir.name}），评估 {len(cases)} 条...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    render = build_renderer()

    json_ok = expr_ok = expr_total = 0
    for c in cases:
        q = c["query"]
        text = f"<|im_start|>user\n{render(q)}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            parsed = json.loads(gen)
            ok = isinstance(parsed, dict) and {"rewritten_query", "filter_expr"} <= parsed.keys()
            json_ok += bool(ok)
            line = f"  模型: {gen}"
            if ok and "expr" in c:
                expr_total += 1
                hit = norm_expr(c["expr"]) == norm_expr(parsed.get("filter_expr"))
                expr_ok += hit
                line += f"\n  expr 期望={c['expr']!r} {'✓' if hit else '✗'}"
        except json.JSONDecodeError:
            line = f"  ✗ 非法JSON: {gen[:80]}"
        print(f"\n[{q}]\n{line}")

    print(f"\n=== 合法 JSON: {json_ok}/{len(cases)}", end="")
    if expr_total:
        print(f" | expr 命中: {expr_ok}/{expr_total}", end="")
    print(" ===")


if __name__ == "__main__":
    main()
