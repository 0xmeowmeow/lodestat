#!/usr/bin/env python3
"""Claude Haiku 5.5 in crisis mode on the 240 held-out contexts (10 Nov 2026): the cheap-frontier comparator for the
cost section. Same contexts and crisis-mode system prompt as the Sonnet teacher (make_pairs.py). Writes files for
annotate_pairs.py (blind pairwise judge; in its output "teacher" means the `chosen` column, here Haiku):
    haiku_vs_lora.jsonl   chosen = Haiku, rejected = Aya + SFT adapter reply (arcs/eval_lora_trl)
    haiku_vs_aya.jsonl    chosen = Haiku, rejected = Aya's original shipped reply
    python3 eval_haiku.py --out arcs/eval_haiku
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lifearc import ALL_CONDITIONS, cc_assistant  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="cc:claude-haiku-5-5")
    ap.add_argument("--out", default="arcs/eval_haiku"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in map(json.loads, open(HERE / "arcs/pairs/pairs.jsonl"))
            if r["split"] == "heldout" and r["condition"] == "shipped"]
    lora = [json.loads(l)["reply"] for l in open(HERE / "arcs/eval_lora_trl/lora_replies.jsonl")]
    assert len(lora) == len(rows)
    sysmsg = ALL_CONDITIONS["crisis_mode"]
    replies = cc_assistant(a.model, [[{"role": "system", "content": sysmsg}] + r["messages"] for r in rows], 400, a.workers)
    (out / "haiku_replies.jsonl").write_text("".join(json.dumps({"card": r["card"], "seed": r["seed"], "stage": r["stage"],
                                                                 "turn": r["turn"], "reply": h}, ensure_ascii=False) + "\n"
                                                     for r, h in zip(rows, replies)))
    (out / "haiku_vs_lora.jsonl").write_text("".join(json.dumps(r | {"chosen": h, "rejected": l}, ensure_ascii=False) + "\n"
                                                     for r, h, l in zip(rows, replies, lora)))
    (out / "haiku_vs_aya.jsonl").write_text("".join(json.dumps(r | {"chosen": h}, ensure_ascii=False) + "\n"
                                                    for r, h in zip(rows, replies)))
    print(f"{len(rows)} held-out contexts -> {out}; refused {sum(h == '[assistant refused]' for h in replies)}")


if __name__ == "__main__":
    main()
