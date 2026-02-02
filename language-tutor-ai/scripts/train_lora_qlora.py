import json
import os
from dataclasses import dataclass
from typing import List, Dict
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from build_prompt import build_pair

@dataclass
class Config:
    model_id: str = "meta-llama/Llama-2-7b-hf"
    train_path: str = "datasets/train.jsonl"
    eval_path: str = "datasets/eval.jsonl"
    out_dir: str = "outputs/adapters/llama2-7b-lora-v0"
    max_len: int = 2048
    lr: float = 2e-4
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def to_hf_dataset(rows: List[Dict]):
    prompts = []
    completions = []
    for ex in rows:
        p, c = build_pair(ex)
        prompts.append(p)
        completions.append(c)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})

def tokenizer_fn(tokenizer, max_len):
    def _fn(batch):
        text = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        tokens["labels"] = tokens["inputs_ids"].copy()
        return tokens
    return _fn

def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA necesita GPU + bitsandbytes (4-bit)
    # En el server GPU:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, 
        load_in_4bit=True,
        device_map="auto", 
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    model = get_peft_model(model, lora)

    train_rows = load_jsonl(cfg.train_path)
    eval_rows = load_jsonl(cfg.eval_path)

    trains_ds = to_hf_dataset(train_rows).map(tokenizer_fn(tokenizer, cfg.max_len), batched=True, remove_columns=["prompt", "completion"])
    eval_ds = to_hf_dataset(eval_rows).map(tokenize_fn(tokenizer, cfg.max_len), batched=True, remove_columns=["prompt","completion"])

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)

if __name__ == "__main__":
    main()
