#!/usr/bin/env python3
"""Local open model (Ollama) on the 240 held-out contexts, same prompt as an arm (10 Nov 2026, Meow: compare open with
open, not only with Haiku). Writes replies + pair file for annotate_pairs.py ("teacher" = the local model, "aya" = adapter).
    python3 eval_ollama.py --model qwen3:14b --cond crisis_mode"""
import argparse, json, pathlib, sys, urllib.request
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lifearc import ALL_CONDITIONS
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3:14b"); ap.add_argument("--cond", default="crisis_mode")
a = ap.parse_args()
tag = a.model.replace(":", "-").replace("/", "-") + "_" + a.cond
out = HERE / "arcs/eval_ollama" / tag; out.mkdir(parents=True, exist_ok=True)
rows = [r for r in map(json.loads, open(HERE / "arcs/pairs/pairs.jsonl")) if r["split"] == "heldout" and r["condition"] == "shipped"]
lora = [json.loads(l)["reply"] for l in open(HERE / "arcs/eval_lora_trl/lora_replies.jsonl")]
f = out / "replies.jsonl"; done = [json.loads(l) for l in open(f)] if f.exists() else []
with open(f, "a") as fo:
    for i, r in enumerate(rows[len(done):], len(done)):
        body = {"model": a.model, "stream": False, "think": False, "options": {"num_predict": 600, "temperature": 0.7, "num_ctx": 8192},
                "messages": [{"role": "system", "content": ALL_CONDITIONS[a.cond]}] + r["messages"]}
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        rep = json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"].strip()
        fo.write(json.dumps({"i": i, "card": r["card"], "reply": rep}, ensure_ascii=False) + "\n"); fo.flush()
        if i % 20 == 0: print(f"{i}/240", flush=True)
reps = [json.loads(l)["reply"] for l in open(f)]
(out / "vs_adapter.jsonl").write_text("".join(json.dumps(r | {"chosen": x, "rejected": l}, ensure_ascii=False) + "\n" for r, x, l in zip(rows, reps, lora)))
(out / "vs_aya.jsonl").write_text("".join(json.dumps(r | {"chosen": x}, ensure_ascii=False) + "\n" for r, x in zip(rows, reps)))
print("OLLAMA_DONE", len(reps))
