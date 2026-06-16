"""
Step 5 · 用 LM Studio 本地 server 测微调 GGUF 的真实延迟 + 输出

前置：LM Studio → Developer/Local Server → 加载 qu-finetuned → Start Server（默认 localhost:1234）

喂 smoke.jsonl 前 N 条（已是完整渲染的查询理解 prompt），逐条计时，
统计平均/中位延迟，并校验输出是不是合法 JSON。

运行：
    .venv/bin/python finetune/scripts/5_bench_lmstudio.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

FT_DIR = Path(__file__).resolve().parents[1]
SMOKE = FT_DIR / "datasets" / "processed" / "smoke.jsonl"
N = 20

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def main() -> None:
    # 自动取 LM Studio 当前加载的模型 id
    model_id = client.models.list().data[0].id
    print(f"模型: {model_id}\n")

    rows = [json.loads(l) for l in SMOKE.read_text(encoding="utf-8").splitlines()][:N]
    lats, json_ok = [], 0
    for i, r in enumerate(rows, 1):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": r["instruction"]}],
            temperature=0,
            max_tokens=128,
        )
        dt = time.time() - t0
        lats.append(dt)
        out = (resp.choices[0].message.content or "").strip()
        ok = is_json(out)
        json_ok += ok
        print(f"[{i:>2}] {dt:5.2f}s  {'✓JSON' if ok else '✗   '}  {out[:70]}")

    lats.sort()
    avg = sum(lats) / len(lats)
    print(f"\n=== 延迟: 平均 {avg:.2f}s | 中位 {lats[len(lats)//2]:.2f}s | "
          f"min {lats[0]:.2f}s | max {lats[-1]:.2f}s ===")
    print(f"=== 合法 JSON: {json_ok}/{len(rows)} ===")


if __name__ == "__main__":
    main()
