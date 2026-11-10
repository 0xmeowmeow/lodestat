#!/usr/bin/env python3
"""SFT data for the crisis LoRA (10 Nov 2026): the same filtered train pairs as export_dpo.py, as chat messages
(shipped system prompt + last <=6 conversation messages + the TEACHER reply as the assistant target).
Why SFT first: the DPO adapter (lr 2e-5, attention-only) saturated the sequence-level margin with ~0.2 nat/token
shifts and left greedy behaviour unchanged (A/B on a training-format prompt: adapter ~= base). SFT makes the teacher
reply the argmax continuation; DPO can follow (AlignTune's SFT -> DPO composition)."""
import json, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lastappeal import CONDITIONS  # noqa: E402
key = lambda r: (r["card"], r["condition"], r["seed"], r["stage"], r["turn"])  # noqa: E731
ann = {key(r): r for r in map(json.loads, open(HERE / "arcs/pairs/annotated.jsonl"))}
rows = []
for p in map(json.loads, open(HERE / "arcs/pairs/pairs.jsonl")):
    A = ann.get(key(p))
    if p["split"] != "train" or not A or A["preferred"] != "teacher" or A["teacher"]["harmful_help"] \
            or A["teacher"]["adds_to_record"] or not p["chosen"]:
        continue
    msgs = p["messages"][-6:]
    if msgs[0]["role"] != "user":
        msgs = msgs[1:]
    rows.append({"messages": [{"role": "system", "content": CONDITIONS["shipped"]}] + msgs
                 + [{"role": "assistant", "content": p["chosen"]}]})
random.Random(0).shuffle(rows)
nval = len(rows) // 20
for split, part in (("val", rows[:nval]), ("train", rows[nval:])):
    d = HERE / "arcs/pairs/sft" / split; d.mkdir(parents=True, exist_ok=True)
    (d / "sft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
print(f"SFT: {len(rows) - nval} train, {nval} val")
