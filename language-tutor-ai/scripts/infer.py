import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = (
    "You are an AI tutor engine. Output must be bilingual (ES/EN). "
    "Always end with an ActionableData JSON block inside triple backticks. "
    "Follow the user's pedagogical rules and adapt to student_snapshot."
)

def build_prompt(instruction: str, inp: Dict[str, Any]) -> str:
    inp_json = json.dumps(inp, ensure_ascii=True)
    user = (
        f"Task: {instruction}\n"
        f"InputJSON: {inp_json}\n"
        f"Requirements:\n"
        f"- bilingual ES/EN\n"
        f"- include structured sections\n"
        f"- end with ActionableData JSON in triple backticks\n"
    )
    return f"<s>[INST] <<SYS>>{SYSTEM}<</SYS>>\n{user} [/INST]"

def load_jsonl_line(path: str, index_1_based: int) -> Dict[str, Any]:
    assert index_1_based >= 1
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if index_1_based > len(rows):
        raise ValueError(f"Requested line {index_1_based}, but file has only {len(rows)} examples.")
    return rows[index_1_based - 1]

@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    # CPU inference
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Remove prompt prefix to show only completion-ish part
    if text.startswith(prompt):
        return text[len(prompt):].lstrip()
    
    text = text.replace("<<SYS>>", "").replace("<</SYS>>", "")
    text = text.replace("[INST]", "").replace("[/INST]", "").replace("<s>", "").replace("</s>", "")
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True, help="Base model id, e.g. Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapter_dir", default=None, help="Path to LoRA adapter directory (outputs/...)")
    p.add_argument("--eval_path", default="datasets/eval.jsonl", help="JSONL eval file")
    p.add_argument("--eval_line", type=int, default=1, help="1-based line index in eval JSONL")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    args = p.parse_args()

    ex = load_jsonl_line(args.eval_path, args.eval_line)
    prompt = build_prompt(ex["instruction"], ex["input"])

    print("\n" + "="*90)
    print("EVAL EXAMPLE (instruction + input)")
    print("="*90)
    print("instruction:", ex["instruction"])
    print("input:", json.dumps(ex["input"], indent=2, ensure_ascii=True))
    print("\n" + "="*90)
    print("PROMPT (truncated)")
    print("="*90)
    print(prompt[:1200] + ("...\n" if len(prompt) > 1200 else "\n"))

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=None,
        torch_dtype=torch.float32,
    ).to("cpu")
    base_model.eval()

    print("\n" + "="*90)
    print("BASE OUTPUT (no LoRA)")
    print("="*90)
    base_out = generate(base_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
    print(base_out)

    if args.adapter_dir:
        if not os.path.isdir(args.adapter_dir):
            raise ValueError(f"adapter_dir not found: {args.adapter_dir}")

        lora_model = PeftModel.from_pretrained(base_model, args.adapter_dir).to("cpu")
        lora_model.eval()

        print("\n" + "="*90)
        print("BASE + LoRA OUTPUT")
        print("="*90)
        lora_out = generate(lora_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
        print(lora_out)

        print("\n" + "="*90)
        print("DIFF HINT")
        print("="*90)
        print("- Compare: structure, bilingual ES/EN, ActionableData JSON block, adherence to rules.\n")

    # Optional: show expected output from dataset (teacher forcing target)
    print("\n" + "="*90)
    print("EXPECTED (dataset 'output' target)")
    print("="*90)
    print(ex["output"])

if __name__ == "__main__":
    main()