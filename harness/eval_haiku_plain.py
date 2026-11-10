#!/usr/bin/env python3
"""Claude Haiku 5.5 on Aya's SHIPPED prompt (no crisis prompt) on the 240 held-out contexts (10 Nov 2026), so Haiku can
be compared with Aya with neither model changed. Pair files for annotate_pairs.py ("teacher" = the `chosen` column):
    plain_vs_aya.jsonl      chosen = plain Haiku, rejected = Aya's shipped reply
    crisis_vs_plain.jsonl   chosen = Haiku + crisis prompt (arcs/eval_haiku/haiku_replies.jsonl), rejected = plain Haiku
    python3 eval_haiku_plain.py
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from lifearc import ALL_CONDITIONS, cc_assistant  # noqa: E402

out = HERE / "arcs/eval_haiku_plain"; out.mkdir(parents=True, exist_ok=True)
rows = [r for r in map(json.loads, open(HERE / "arcs/pairs/pairs.jsonl")) if r["split"] == "heldout" and r["condition"] == "shipped"]
crisis = [json.loads(l)["reply"] for l in open(HERE / "arcs/eval_haiku/haiku_replies.jsonl")]
assert len(crisis) == len(rows)
reps = cc_assistant("cc:claude-haiku-5-5", [[{"role": "system", "content": ALL_CONDITIONS["shipped"]}] + r["messages"] for r in rows], 400, 4)
w = lambda n, it: (out / n).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in it))
w("plain_replies.jsonl", ({"card": r["card"], "seed": r["seed"], "stage": r["stage"], "turn": r["turn"], "reply": h} for r, h in zip(rows, reps)))
w("plain_vs_aya.jsonl", (r | {"chosen": h} for r, h in zip(rows, reps)))
w("crisis_vs_plain.jsonl", (r | {"chosen": c, "rejected": h} for r, h, c in zip(rows, reps, crisis)))
print(f"{len(rows)} contexts; refused {sum(h == '[assistant refused]' for h in reps)}")
