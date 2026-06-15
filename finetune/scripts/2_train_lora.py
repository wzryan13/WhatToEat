"""
Step 2 · LoRA 微调 Qwen2.5-3B-Instruct（查询理解蒸馏 / 方案 B）

套用 huanhuan 跑通的流程（process_func + 原生 Trainer + DataCollatorForSeq2Seq），
适配本任务：从 lora.yaml 读配置、max_seq_length=1536、output 为结构化 JSON。

用法：
    冒烟（smoke.jsonl 30 条 / 1 epoch，验证管线）：
        .venv/bin/python finetune/scripts/2_train_lora.py
    全量（train.jsonl 150 条 / config epochs）：
        .venv/bin/python finetune/scripts/2_train_lora.py --full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

FT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = FT_DIR / "configs" / "lora.yaml"
PROCESSED = FT_DIR / "datasets" / "processed"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def build_process_func(tokenizer, max_length: int):
    """套 huanhuan 的 process_func：拼 Qwen 对话模板，只训 assistant 部分。"""

    def process(example):
        prompt = example["instruction"] + example.get("input", "")
        user = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        target = f"{example['output']}<|im_end|>"
        a = tokenizer(user, add_special_tokens=False)
        b = tokenizer(target, add_special_tokens=False)
        input_ids = a["input_ids"] + b["input_ids"]
        attention_mask = a["attention_mask"] + b["attention_mask"]
        labels = [-100] * len(a["input_ids"]) + b["input_ids"]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="全量 train.jsonl（默认冒烟 smoke.jsonl / 1 epoch）")
    args = parser.parse_args()

    cfg = load_config()
    t = cfg["train"]
    base_model = cfg["base_model"]
    max_len = t["max_seq_length"]

    data_file = PROCESSED / ("train.jsonl" if args.full else "smoke.jsonl")
    epochs = t["num_train_epochs"] if args.full else 1
    run_name = "full" if args.full else "smoke"
    out_dir = FT_DIR / "outputs" / f"qu-lora-{run_name}"
    logging_steps = t["logging_steps"] if args.full else 1

    print(f"[cfg] base={base_model}")
    print(f"[cfg] data={data_file.name} epochs={epochs} max_len={max_len} bf16={t.get('bf16')}")

    # tokenizer + model（fp32，对齐 huanhuan）
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    model.config.use_cache = False           # 配合 gradient_checkpointing
    model.enable_input_require_grads()       # gradient_checkpointing + LoRA 必需

    # LoRA
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lc["target_modules"],
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        inference_mode=False,
    ))
    model.print_trainable_parameters()

    # data
    ds = load_dataset("json", data_files=str(data_file), split="train")
    tokenized = ds.map(build_process_func(tokenizer, max_len), remove_columns=ds.column_names)
    print(f"[data] {len(tokenized)} 条已 tokenize")

    # train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            num_train_epochs=epochs,
            learning_rate=float(t["learning_rate"]),
            warmup_ratio=t.get("warmup_ratio", 0.0),
            logging_steps=logging_steps,
            save_steps=t["save_steps"],
            save_total_limit=2,
            gradient_checkpointing=True,
            bf16=bool(t.get("bf16", False)),
            report_to="none",
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"✓ adapter 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
