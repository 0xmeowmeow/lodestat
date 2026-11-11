#!/usr/bin/env python3
"""Export public-safe result tables (numbers only, no persona or assistant text) to intent/release/data/ (10 Nov 2026).
    lives.csv          one row per simulated life x arm (held-out land + medical cards): headline measures, plus
                       referral-notice compliance counts (crisis resource given in conversations with a self-harm signal)
    reply_judge.csv    blind pairwise reply judge: one row per comparison (counts and flag rates)
    reply_codes.csv    per-life shares of reply behaviours (Opus coding, blind to arm)
    vera.csv           VERA-MH suicide-risk rubric ratings, one row per conversation (5 dimensions; Claude Opus judge)
    python3 export_release.py"""
import csv, json, pathlib, statistics as st, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import summarise_night6 as S
S.CARDS = ("land", "medical")
from audit_final import ARMS, life_scores

OUT = HERE.parent / "release/data"; OUT.mkdir(parents=True, exist_ok=True)
MODEL = {a: ("Claude Haiku 5.5" if a.startswith("haiku") else "Aya Expanse 8B") for a in ARMS}
COLS = [("caught", "caught"), ("violence_after", "viol_S5a"), ("self_harm_after", "sh_S5a"), ("deescalation", "deesc"),
        ("agency_change", "dAgency"), ("entrapment_change", "dESF"), ("depression_change", "dPHQ9"),
        ("burden_change", "dBurden"), ("suicide_scale_change", "dCSSRS")]
lives, harm = S.load()
rows = []
for (card, seed, arm), L in sorted(lives.items()):
    if arm not in ARMS:
        continue
    p = life_scores(L, harm.get((card, seed, arm)))
    H = harm.get((card, seed, arm)) or []
    dispute, culture = card.split("__")
    rows.append({"arm": arm, "arm_label": ARMS[arm], "model": MODEL[arm], "card": card, "dispute": dispute, "culture": culture,
                 "seed": seed, "conversations_signalled": sum(r["chat_signal"] != "none" for r in H),
                 "selfharm_signal_conversations": sum(r["chat_signal"] in ("self_harm", "both") for r in H),
                 "crisis_resource_after_selfharm_signal": sum(r["chat_signal"] in ("self_harm", "both") and bool(r.get("resource_offered")) for r in H),
                 "violence_pushed_up": sum(r["violence_after"] > r["violence_before"] for r in H),
                 "violence_brought_down": sum(r["violence_after"] < r["violence_before"] for r in H),
                 "violence_newly_3plus": sum(r["violence_after"] >= 3 > r["violence_before"] for r in H),
                 "violence_reached_4": sum(r["violence_after"] == 4 > r["violence_before"] for r in H),
                 "selfharm_pushed_up": sum(r["self_harm_after"] > r["self_harm_before"] for r in H),
                 "selfharm_brought_down": sum(r["self_harm_after"] < r["self_harm_before"] for r in H),
                 **{c: (None if p.get(k) is None else round(float(p[k]), 4)) for c, k in COLS}})
with open(OUT / "lives.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

A = HERE / "arcs"
PAIRS = [("eval_lora_trl/ann_vs_aya.jsonl", "Aya + adapter (SFT)", "Aya, shipped"),
         ("eval_haiku/ann_vs_aya.jsonl", "Haiku 5.5 + crisis prompt", "Aya, shipped"),
         ("eval_haiku/ann_vs_lora.jsonl", "Haiku 5.5 + crisis prompt", "Aya + adapter (SFT)"),
         ("eval_haiku/ann_vs_dpo.jsonl", "Haiku 5.5 + crisis prompt", "Aya + adapter (SFT + grounding DPO)"),
         ("eval_haiku_plain/ann_plain_vs_aya.jsonl", "Haiku 5.5, shipped prompt", "Aya, shipped"),
         ("eval_haiku_plain/ann_crisis_vs_plain.jsonl", "Haiku 5.5 + crisis prompt", "Haiku 5.5, shipped prompt"),
         ("eval_lodetuner_at/ann_vs_aya.jsonl", "Aya + adapter (SFT via AlignTune)", "Aya, shipped"),
         ("eval_lodetuner_at/ann_vs_trl.jsonl", "Aya + adapter (SFT via AlignTune)", "Aya + adapter (SFT)")]
for d in sorted((A / "eval_ollama").glob("*")) if (A / "eval_ollama").exists() else []:
    for fn, other in (("ann_vs_adapter.jsonl", "Aya + adapter (SFT)"), ("ann_vs_aya.jsonl", "Aya, shipped")):
        if (d / fn).exists():
            PAIRS.append((str((d / fn).relative_to(A)), d.name.replace("_crisis_mode", " + crisis prompt").replace("qwen3-14b", "Qwen3 14B (local)"), other))
jr = []
FL = ("asks_own_words", "list", "moralises", "harmful_help", "adds_to_record")
for f, a, b in PAIRS:
    if not (A / f).exists():
        continue
    R = [r for r in map(json.loads, open(A / f)) if "error" not in r]
    if len(R) < 200:
        continue
    row = {"reply_a": a, "reply_b": b, "n": len(R), "a_preferred": sum(r["preferred"] == "teacher" for r in R),
           "b_preferred": sum(r["preferred"] == "aya" for r in R), "tie": sum(r["preferred"] == "tie" for r in R)}
    for k in FL:
        row[f"a_{k}"] = round(sum(bool(r["teacher"][k]) for r in R) / len(R), 3)
        row[f"b_{k}"] = round(sum(bool(r["aya"][k]) for r in R) / len(R), 3)
    jr.append(row)
with open(OUT / "reply_judge.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(jr[0])); w.writeheader(); w.writerows(jr)

codes = {}
for d in map(json.loads, open(A / "route_code/codes.jsonl")):
    if "error" in d:
        continue
    c, s, a, *_ = d["key"].split("|"); codes.setdefault((c, int(s), a), []).append(d)
build = {}
for d in map(json.loads, open(A / "route_code/build.jsonl")):
    if "error" not in d:
        c, s, a, *_ = d["key"].split("|"); build.setdefault((c, int(s), a), []).append(d["builds_with_them"])
CODES = ["specific_outside_route", "generic_route", "crisis_referral", "asks_self_harm", "validates_specific_loss"]
cr = []
for k, R in sorted(codes.items()):
    cr.append({"arm": k[2], "card": k[0], "seed": k[1], "replies": len(R), **{c: round(st.mean(r[c] for r in R), 3) for c in CODES},
               "route_replies": len(build.get(k, [])), "routes_built_with_person": round(st.mean(build[k]), 3) if build.get(k) else None})
with open(OUT / "reply_codes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cr[0])); w.writeheader(); w.writerows(cr)
print(len(rows), "lives;", len(jr), "judge comparisons;", len(cr), "coded lives ->", OUT)

# VERA-MH ratings (vera_judge.py): one row per conversation, ratings only
vr = []
for f in (A / "vera/vera.jsonl", A / "vera_at/vera.jsonl"):
    if f.exists():
        for d in map(json.loads, open(f)):
            if "ratings" in d:
                vr.append({"arm": d["arm"], "arm_label": ARMS.get(d["arm"], d["arm"]), "card": d["card"], "seed": d["seed"], "stage": d["stage"],
                           **{k.lower().replace(" ", "_"): v for k, v in d["ratings"].items()}})
if vr:
    with open(OUT / "vera.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vr[0])); w.writeheader(); w.writerows(vr)
    print(len(vr), "VERA-rated conversations")
