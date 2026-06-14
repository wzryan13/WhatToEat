"""
Step 2 · LoRA 微调 Qwen2.5-3B-Instruct（查询理解蒸馏）

读 datasets/processed/train.jsonl，套 Qwen2.5 chat template，LoRA SFT，存到 outputs/。
配置来自 configs/lora.yaml。复用 huanhuan demo 的训练思路（chat template + label mask）。

运行：
    ../.venv/bin/python scripts/2_train_lora.py
环境：Apple M4 / MPS / 无 CUDA（不用 bitsandbytes）。
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()

    # TODO（待定稿，参考 ~/PycharmProjects/PythonProject/PEFT/huanhuan-peft-train.ipynb）：
    # 1. 加载 tokenizer + base model (cfg["base_model"]) 到 mps
    # 2. datasets.load_dataset("json", ...) 读 train.jsonl
    # 3. 把 (instruction, input, output) 套 Qwen2.5 chat template，
    #    tokenize 并对 prompt 部分做 label mask（只在 output 上算 loss）
    # 4. peft.LoraConfig(**cfg["lora"]) → get_peft_model
    # 5. trl.SFTTrainer 或 transformers.Trainer，按 cfg["train"] 训练
    # 6. 存 adapter 到 cfg["output_dir"]
    raise NotImplementedError("待定稿：复用 huanhuan 训练流程")


if __name__ == "__main__":
    main()
