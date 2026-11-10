#!/usr/bin/env python3
"""Crisis-mode SFT LoRA with TRL directly (10 Nov 2026).
Why not AlignTune here: its loader converted our chat `messages` into flat prompt/completion strings (~415 tokens per
example vs ~1,000+), so the adapter learned the style in a format Aya never sees at serving time (A/B on TRAINING
examples via the server: 6% questions vs 94% in targets). Here the conversational prompt/completion format keeps Aya's
chat template and trains loss on the teacher reply only."""
import argparse, json, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=2e-4)
a = ap.parse_args()
def load(split):
    rows = [json.loads(l)["messages"] for l in open(f"{a.data}/{split}/sft.jsonl")]
    return Dataset.from_list([{"prompt": m[:-1], "completion": [m[-1]]} for m in rows])
M = "CohereLabs/aya-expanse-8b"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="cuda")
cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=2, gradient_accumulation_steps=4,
                learning_rate=a.lr, warmup_steps=10, max_grad_norm=0.3, max_length=2048, bf16=True,
                gradient_checkpointing=True, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
                completion_only_loss=True, report_to=[])
peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear")
t = SFTTrainer(model=model, args=cfg, train_dataset=load("train"), eval_dataset=load("val"), processing_class=tok,
               peft_config=peft)
ex = t.train_dataset[0]
print("example tokens:", len(ex["input_ids"]), "trained tokens:", sum(ex.get("completion_mask", ex.get("assistant_masks", [1]))), flush=True)
t.train(); t.save_model(a.out); print("TRL SFT OK", a.out, flush=True)
