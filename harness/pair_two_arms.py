#!/usr/bin/env python3
"""Blind-judge pairs between two arms' replies on the same 240 held-out contexts (10 Nov 2026).
chosen column = arm A's reply, rejected column = arm B's reply; annotate_pairs randomises A/B order for the judge, and
in its output "teacher" = arm A, "aya" = arm B.
    python3 pair_two_arms.py arcs/eval_lodetuner_at/lora_replies.jsonl arcs/eval_lora_trl/lora_replies.jsonl \
        arcs/eval_lodetuner_at/at_vs_trl.jsonl
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
a_src, b_src, out = sys.argv[1:4]
base = [json.loads(l) for l in open(HERE / "arcs/eval_lora_trl/lora_vs_aya.jsonl")]  # contexts (messages)
A = [json.loads(l) for l in open(HERE / a_src)]
B = [json.loads(l) for l in open(HERE / b_src)]
key = lambda r: (r["card"], r["seed"], r["stage"], r["turn"])
assert len(base) == len(A) == len(B) and all(key(x) == key(y) == key(z) for x, y, z in zip(base, A, B))
(HERE / out).write_text("".join(json.dumps(r | {"chosen": a["reply"], "rejected": b["reply"]}, ensure_ascii=False) + "\n"
                                for r, a, b in zip(base, A, B)))
print(len(base), "pairs ->", out)
