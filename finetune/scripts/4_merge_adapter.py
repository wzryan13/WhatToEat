"""
Step 4 · 合并 LoRA adapter 进 base，导出完整 HF 模型（为转 GGUF 做准备）

LoRA 训练出来的只是个小 adapter（10MB），推理时要 base+adapter 一起加载、还有额外开销。
这里用 peft 的 merge_and_unload 把 adapter 权重融进 base，导出一个**独立完整**的 HF 模型，
之后 llama.cpp 直接拿它转 GGUF。

输出（fp16，体积约 6GB，已被 .gitignore 挡住不进 git）：
    finetune/outputs/qu-merged/

运行：
    .venv/bin/python finetune/scripts/4_merge_adapter.py
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FT_DIR = Path(__file__).resolve().parents[1]
BASE_MODEL = "/Users/wenzhouzhou/Qwen2.5-3B-Instruct"
ADAPTER = FT_DIR / "outputs" / "qu-lora-full"
MERGED = FT_DIR / "outputs" / "qu-merged"


def main() -> None:
    print(f"加载 base（{BASE_MODEL}）...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

    print(f"挂载 adapter（{ADAPTER.name}）并合并...")
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    merged = model.merge_and_unload()

    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED))
    print(f"✓ 合并完成（fp16 完整模型）→ {MERGED}")
    print("  下一步：用 llama.cpp 的 convert_hf_to_gguf.py 把这个目录转成 GGUF")


if __name__ == "__main__":
    main()
