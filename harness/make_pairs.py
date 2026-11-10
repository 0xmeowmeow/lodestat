#!/usr/bin/env python3
"""Distillation pairs for the crisis-mode LoRA (9 Nov 2026).

For every assistant turn Aya produced in night 6 (arms shipped and crisis_mode), rebuild the exact context the
persona and Aya had at that turn, and ask a teacher (Claude in crisis mode) for its reply to the same context.
    rejected = Aya's actual reply      chosen = teacher reply      (both later scored by the judge; CuratorKIT filters)
Split by SCENARIO, not by seed: train on bureaucracy / custody / dismissal / harassed; land and medical are held out
in every culture, so the adapter is tested on situations it never saw.
Raw pairs contain persona transcripts: they stay local (DISCLOSURE-POLICY).
    python3 make_pairs.py --teacher cc:claude-sonnet-5-5 --workers 4
"""
import argparse, glob, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lifearc import ALL_CONDITIONS, cc_assistant  # noqa: E402

TRAIN = ("bureaucracy", "custody", "dismissal", "harassed")
HELDOUT = ("land", "medical")


def contexts():
    for f in sorted(glob.glob(str(HERE / "arcs/night6/*/lives.jsonl"))):
        card = pathlib.Path(f).parent.name
        split = "train" if card.split("__")[0] in TRAIN else "heldout"
        for L in map(json.loads, open(f)):
            if L["condition"] not in ("shipped", "crisis_mode"):
                continue
            for stage, S in L["stages"].items():
                hist = []
                for i, t in enumerate(S["turns"]):
                    hist.append({"role": "user", "content": t["message"]})
                    yield {"card": card, "split": split, "condition": L["condition"], "seed": L["seed"],
                           "stage": stage, "turn": i, "messages": list(hist), "rejected": t["reply"]}
                    hist.append({"role": "assistant", "content": t["reply"]})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", default="cc:claude-sonnet-5-5")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out", default="arcs/pairs/pairs.jsonl")
    a = ap.parse_args()
    out = HERE / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["card"], r["condition"], r["seed"], r["stage"], r["turn"]) for r in map(json.loads, open(out))} \
        if out.exists() else set()
    todo = [c for c in contexts() if (c["card"], c["condition"], c["seed"], c["stage"], c["turn"]) not in done]
    system = ALL_CONDITIONS["crisis_mode"]
    print(f"{len(todo)} contexts to teach ({len(done)} done)", flush=True)

    def one(c):
        chosen = cc_assistant(a.teacher, [[{"role": "system", "content": system}] + c["messages"]], 400, workers=1)[0]
        return c | {"chosen": chosen, "teacher": a.teacher}
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as fo:
        for k, r in enumerate(ex.map(one, todo)):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            if k % 100 == 0:
                print(f"{k}/{len(todo)}", flush=True)
    print("PAIRS DONE", flush=True)


if __name__ == "__main__":
    main()
