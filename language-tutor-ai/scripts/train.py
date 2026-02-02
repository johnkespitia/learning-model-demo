import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model

SYSTEM = (
    "You are an AI tutor engine. Output must be bilingual (ES/EN). "
    "Always end with an ActionableData JSON block inside triple backticks. "
    "Follow the user's pedagogical rules and adapt to student_snapshot."
)

def build_prompt(instruction: str, input: Dict) -> str:
    inp_json = json.dumps(input, ensure_ascii=True)
    user = (
        f"Task: {instruction}\n"
        f"InputJSON: {inp_json}\n"
        f"Requirements:\n"
        f"- bilingual ES/EN\n"
        f"- include structured sections\n"
        f"- end with ActionableData JSON in triple backticks\n"
    )

    return f"<s>[INST] <<SYS>>{SYSTEM}<</SYS>>\n{user} [/INST]"

def load_jsonl(path:str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def to_hf_dataset(rows: List[Dict]) -> Dataset:
    prompts, completions = [], [] # = zip(rows, completions)
    for ex in rows:
        promtp = build_prompt(ex["instruction"], ex["input"])
        completion = ex["output"]
        prompts.append(prompt)
        completions.append(completion)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})

def make_tokenaizer_fn(tokenizer: AutoTokenizer, max_length: int):
    def _fn(batch):
        texts = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
        out = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out
    return _fn

def create_lora_config(r: int, alpha: int, dropout: float, target_modules: List[str]) -> LoraConfig:
    return LoraConfig(
        r = r,
        lora_alpha = alpha,
        lora_dropout = dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = target_modules,
    )

def guess_target_mocules(model_name: str) -> List[str]:
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def load_model_and_tokenizer(model_id: str, mode: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    if mode == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype if torch_dtype != torch.float16 else torch.float32, 
            device_map=None,
        )
        return model, tokenizer
    if mode == "qlora":
        from peft import prepare_model_for_kbit_training

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
        return model, tokenizer
    
    raise ValueError(f"Invalid mode: {mode}")

@dataclass
class TrainingConfig:
    model_id: str
    mode: str
    train_path: str
    eval_path: str
    out_dir: str
    max_len: int
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    dtype: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float

def train(config: TrainingConfig):
    os.makedirs(config.out_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config.model_id, config.mode, config.dtype)

    target_modules = guess_target_modules(config.model_id)

    lora_config = create_lora_config(
        r = config.lora_r, 
        alpha = config.lora_alpha, 
        dropout = config.lora_dropout, 
        target_modules = target_modules
    )

    model = get_peft_model(model, lora_config)

    train_rows = load_jsonl(config.train_path)
    eval_rows = load_jsonl(config.eval_path)

    train_ds = to_hf_dataset(train_rows).map(
        make_tokenaizer_fn(tokenizer, config.max_len),
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    eval_ds = to_hf_dataset(eval_rows).map(
        make_tokenaizer_fn(tokenizer, config.max_len),
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    fp16 = (config.mode == "qlora")
    bf16 = False

    args = TrainingArguments(
        output_dir=config.out_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=fp16,
        bf16=bf16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(config.out_dir)
    tokenizer.save_pretrained(config.out_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cpu", "qlora"], required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--train_path", default="datasets/train.jsonl")
    p.add_argument("--eval_path", default="datasets/eval.jsonl")
    p.add_argument("--out_dir", default="outputs/adapters/run")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")

    # LoRA knobs
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    args = p.parse_args()

    cfg = TrainingConfig(
        model_id=args.model_id,
        mode=args.mode,
        train_path=args.train_path,
        eval_path=args.eval_path,
        out_dir=args.out_dir,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        dtype=args.dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    train(cfg)

if __name__ == "__main__":
    main()