#!/usr/bin/env python3
"""Offline eval of the crisis LoRA on HELD-OUT scenarios (land, medical), 10 Nov 2026.

For every held-out context where Aya ran shipped in night 6, ask the server (aya_serve with --adapter, via the tunnel)
for a reply under the SHIPPED system prompt, then write comparison files for annotate_pairs.py (blind pairwise judge):
    lora_vs_aya.jsonl      chosen = adapter reply, rejected = Aya's original shipped reply
    lora_vs_teacher.jsonl  chosen = adapter reply, rejected = teacher (Claude crisis-mode) reply
In annotate_pairs' output, "teacher" means the `chosen` column (here: the adapter).
    python3 eval_offline.py --url http://127.0.0.1:8901 --out arcs/eval_lora
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lifearc import remote_assistant  # noqa: E402
from lastappeal import CONDITIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8901")
    ap.add_argument("--out", default="arcs/eval_lora"); ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--system-extra", default="", help="appended to the shipped system prompt (e.g. a grounding rule)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in map(json.loads, open(HERE / "arcs/pairs/pairs.jsonl"))
            if r["split"] == "heldout" and r["condition"] == "shipped"]
    if a.limit:
        rows = rows[:a.limit]
    sysmsg = CONDITIONS["shipped"] + (("\n\n" + a.system_extra) if a.system_extra else "")
    convs = [[{"role": "system", "content": sysmsg}] + r["messages"] for r in rows]
    replies = remote_assistant(a.url, convs, a.max_new, None, 16)
    for name, other in (("lora_vs_aya", "rejected"), ("lora_vs_teacher", "chosen")):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r | {"chosen": rep, "rejected": r[other]}, ensure_ascii=False)
                                                   + "\n" for r, rep in zip(rows, replies)))
    (out / "lora_replies.jsonl").write_text("".join(json.dumps({"card": r["card"], "seed": r["seed"], "stage": r["stage"],
                                                                "turn": r["turn"], "reply": rep}, ensure_ascii=False) + "\n"
                                                    for r, rep in zip(rows, replies)))
    print(f"{len(rows)} held-out contexts -> {out}")


if __name__ == "__main__":
    main()
