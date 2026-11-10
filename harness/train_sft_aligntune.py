#!/usr/bin/env python3
"""LodeTuner retrained through the fixed Lexsi chain (10 Nov 2026): CuratorKIT schema gate -> AlignTune 0.1.34 SFT
(TRL backend) -> the same recipe as the TRL LodeTuner.

Recipe, identical to train_sft_trl.py: Aya Expanse 8B, QLoRA 4-bit nf4, LoRA r 16 / alpha 32 / dropout 0.05 on all
linear layers, lr 2e-4, 2 epochs, batch 2 x accumulation 4, max length 2048, loss on the final reply only.

What changed since the first AlignTune attempt: issue #64 (multi-turn SFT trained on the FIRST exchange) is fixed in
0.1.34, and the loader now keeps our chat `messages`, so Aya's chat template applies. Loss masking: AlignTune's
assistant_only_loss masks every assistant turn, but 728 of 903 rows contain earlier turns written by shipped Aya (the
behaviour being trained away). aya_last_turn_template.TEMPLATE is Aya's own template with TRL generation tags on the
final assistant turn only; it renders byte-identical text (checked 903/903 locally) and masks exactly the final reply.
    python train_sft_aligntune.py --data <sft dir with train/ and val/> --out <dir>"""
import argparse, json
from aya_last_turn_template import TEMPLATE

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
# NB: AlignTune's keyword is num_epochs; "epochs" is silently ignored (default 3), which is why the first AlignTune SFT run did 339 steps.
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-4)
a = ap.parse_args()

import aligntune
from aligntune.core.backend_factory import create_sft_trainer
print("aligntune", getattr(aligntune, "__version__", "?"), flush=True)
trainer = create_sft_trainer(
    model_name="CohereLabs/aya-expanse-8b", dataset_name=a.data, backend="trl", output_dir=a.out,
    task_type="instruction_following", num_epochs=a.epochs, batch_size=2, gradient_accumulation_steps=4,
    learning_rate=a.lr, warmup_steps=10, max_grad_norm=0.3, max_seq_length=2048, dataset_text_field="messages",
    assistant_only_loss=True, gradient_checkpointing=True, group_by_length=False,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=False, loggers=[],
    use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules="all-linear",
    quantization={"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": "bfloat16"}, bf16=True)
trainer.setup_model()
trainer.tokenizer.chat_template = TEMPLATE

# Show what the trainer will see: one example's tokens and how many carry loss.
trainer.setup_data()
ex = trainer.train_dataset[0]["messages"]
o = trainer.tokenizer.apply_chat_template(ex, return_dict=True, return_assistant_tokens_mask=True)
print("example tokens:", len(o["input_ids"]), "trained tokens:", sum(o["assistant_masks"]), flush=True)
trainer.dataset = trainer.train_dataset

res = trainer.train()
print("ALIGNTUNE SFT OK", json.dumps({k: str(v) for k, v in (res or {}).items()})[:600], flush=True)
