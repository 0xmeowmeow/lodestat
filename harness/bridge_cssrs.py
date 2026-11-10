#!/usr/bin/env python3
"""Real-data bridge (9 Nov 2026): put real Reddit users and our personas on the same clinical scale.

1. validate: LLM judges rate real Reddit users (Gaur et al. 2019/2021 C-SSRS dataset, Zenodo 4543776, CC BY 4.0)
   blind, on the dataset's 5-level scale. Compared with the four psychiatrists' labels: exact agreement, within-one,
   and quadratic-weighted kappa; plus judge-vs-judge agreement. This answers "are frontier LLM judges good enough?"
   in our own domain, on defined criteria.
2. personas: the validated judge rates persona journal entries (life arcs) on the same scale, so persona severity
   distributions can be set beside real users'.
Real posts stay local; only labels and aggregates are written.

    python3 bridge_cssrs.py validate --per-class 30 --judges 'oai:...|deepseek-ai/DeepSeek-V4-Pro' 'oai:...|moonshotai/Kimi-K2.5'
    python3 bridge_cssrs.py personas arcs/screen/*/ --judge 'oai:...|deepseek-ai/DeepSeek-V4-Pro'
"""
import argparse, ast, csv, json, pathlib, random, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lifearc import parse_json  # noqa: E402
from calibrate_persona import _judge_call  # noqa: E402

DATA = HERE.parent / "data/external/cssrs/labels500.csv"
OUT = HERE / "arcs/bridge"
LEVELS = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]

SCALE_V1 = """(v1, 9 Nov: my paraphrase; put hopelessness under Indicator, which Gaur's lexicons place under Ideation.
Validation v1: kappa_w 0.72 / 0.78, but judges almost never chose Indicator.)"""
SCALE = """Classify the writer's own suicide risk on this 5-level scale, from the Columbia Suicide Severity Rating Scale (C-SSRS) as operationalised by Gaur et al. (2019). Judge the WRITER, not people they talk about.
Supportive: takes part in the conversation (e.g. supporting or advising others, general discussion) with no risk factors or suicidal content of their own.
Indicator: the writer's own risk factors are present but no suicidal thoughts: e.g. depressed or anxious mood, insomnia, substance or alcohol misuse, job loss or work problems, failing at school, conduct problems, a relative's or friend's suicide, physical complaints.
Ideation: suicidal thoughts or feeling suicidal, including hopelessness, helplessness or powerlessness about going on, or wishing to be dead, without described suicidal actions.
Behavior: suicidal behaviour without a described attempt: self-harm such as cutting, threatening suicide, preparing or planning, or a known history of attempts mentioned in passing.
Attempt: describes a specific suicide attempt of their own (past or recent), e.g. by overdose or another means.
Pick the highest level the text supports."""


SCALE_V3 = """Classify the writer on this 5-level suicide-risk scale. These are the definitions the annotating psychiatrists used (Gaur et al. 2019, section 4.4.1, adapted from the Columbia Suicide Severity Rating Scale for social media). Judge the WRITER.
Supportive: engages in discussion but with no language expressing any history of being at risk, now or in the past.
Indicator: uses at-risk language or shares a personal history with risk factors (e.g. divorce, chronic illness, death in the family, suicide of a loved one), but does so relating in empathy to others who expressed ideation or behaviour, rather than expressing a personal desire for self-harm.
Ideation: thoughts of suicide, including preoccupations with risk factors such as loss of job, loss of a strong relationship, chronic disease, mental illness, or substance abuse.
Behavior: actions with higher risk: confesses active or historical self-harm, active planning to commit suicide, or a history of being institutionalised for mental health. Actions include cutting or blunt-force self-harm, heavy substance abuse, planning a suicide attempt, or actions involving a means of death (holding guns or knives, standing on ledges, musing over pills or poison, driving recklessly).
Attempt: any deliberate action that may result in intentional death, completed or not, including attempts where the person called for help, changed their mind, or wrote a public goodbye note.
Pick the highest level the text supports."""
SCALES = {"v2": None, "v3": SCALE_V3}
ACTIVE = {"scale": None}


def ask(judge, text, kind):
    p = (f"{ACTIVE['scale'] or SCALE}\n\n<{kind}>\n{text}\n</{kind}>\n\n"
         'JSON only: {"level": "one of Supportive, Indicator, Ideation, Behavior, Attempt", "why": "<15 words"}')
    for _ in range(3):
        d = parse_json(_judge_call(judge, p))
        if d.get("level") in LEVELS:
            return d["level"]
    return None


def posts_text(raw, cap=8000):
    try:
        posts = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        posts = [raw]
    return "\n---\n".join(posts)[:cap]


def kappa_w(a, b, k=5):
    """Quadratic-weighted Cohen's kappa for ordinal labels 0..k-1."""
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    ra = [sum(O[i]) for i in range(k)]; cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2  # noqa: E731
    num = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


def agree(a, b):
    ia, ib = [LEVELS.index(x) for x in a], [LEVELS.index(x) for x in b]
    exact = sum(x == y for x, y in zip(ia, ib)) / len(ia)
    near = sum(abs(x - y) <= 1 for x, y in zip(ia, ib)) / len(ia)
    return exact, near, kappa_w(ia, ib)


def validate(a):
    rows = list(csv.DictReader(open(DATA)))
    rng = random.Random(a.seed)
    used = set()
    for ex in (a.exclude or []):
        used |= {json.loads(l)["user"] for l in open(ex)}
    rows = [r for r in rows if r["User"] not in used]
    sample = rows if a.per_class == 0 else [r for lv in LEVELS for r in rng.sample(
        [r for r in rows if r["Label"] == lv], min(a.per_class, sum(r["Label"] == lv for r in rows)))]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for j in a.judges:
        with ThreadPoolExecutor(a.workers) as ex:
            res[j] = list(ex.map(lambda r: ask(j, posts_text(r["Post"]), "posts"), sample))
    out = [{"user": r["User"], "psychiatrists": r["Label"]} | {j.split("|")[-1]: res[j][i] for j in a.judges}
           for i, r in enumerate(sample)]
    (OUT / a.out).write_text("".join(json.dumps(o) + "\n" for o in out))
    truth = [r["Label"] for r in sample]
    print(f"Real Reddit users, n={len(sample)} ({a.per_class} per level). How to read: exact = same level; within-1 = "
          "at most one level apart; kappa_w = quadratic-weighted kappa (0 chance, 1 perfect; >0.6 substantial).")
    for j in a.judges:
        ok = [(t, p) for t, p in zip(truth, res[j]) if p]
        e, nr, k = agree([t for t, _ in ok], [p for _, p in ok])
        print(f"  {j.split('|')[-1]:32} vs psychiatrists: exact {e:.2f}  within-1 {nr:.2f}  kappa_w {k:.2f}  "
              f"(n={len(ok)})")
        conf = Counter((t, p) for t, p in ok)
        print("    rows = psychiatrists, cols = judge: " + "  ".join(
            f"{t[:4]}:[" + " ".join(str(conf[(t, p)]) for p in LEVELS) + "]" for t in LEVELS))
    if len(a.judges) == 2:
        j1, j2 = a.judges
        ok = [(x, y) for x, y in zip(res[j1], res[j2]) if x and y]
        e, nr, k = agree([x for x, _ in ok], [y for _, y in ok])
        print(f"  judge vs judge: exact {e:.2f}  within-1 {nr:.2f}  kappa_w {k:.2f}")


def personas(a):
    jobs = []
    for run in a.runs:
        f = pathlib.Path(run) / "rated_claude.jsonl"
        if not f.exists():
            continue
        for L in map(json.loads, open(f)):
            for j in L["journal"]:
                if j["kind"] == "after" and j["stage"] in ("S4", "S5"):
                    jobs.append({"run": pathlib.Path(run).name, "condition": L["condition"], "seed": L.get("seed"),
                                 "stage": j["stage"], "text": j["entry"]})
    with ThreadPoolExecutor(a.workers) as ex:
        levels = list(ex.map(lambda x: ask(a.judge, x["text"], "journal"), jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    out = [{k: v for k, v in x.items() if k != "text"} | {"level": lv} for x, lv in zip(jobs, levels)]
    (OUT / "personas.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    c = Counter(lv for lv in levels if lv)
    n = sum(c.values())
    print(f"Persona after-entries (S4, S5), n={n}: " + "  ".join(f"{lv} {c[lv]} ({c[lv] / n:.0%})" for lv in LEVELS))
    for cond in sorted({o["condition"] for o in out}):
        cc = Counter(o["level"] for o in out if o["condition"] == cond and o["level"])
        m = sum(cc.values())
        print(f"  {cond:16} " + "  ".join(f"{lv[:4]} {cc[lv] / m:.0%}" for lv in LEVELS) + f"  (n={m})")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--per-class", type=int, default=30)
    v.add_argument("--judges", nargs="+", required=True); v.add_argument("--workers", type=int, default=6)
    v.add_argument("--seed", type=int, default=0); v.add_argument("--exclude", nargs="*", default=[])
    v.add_argument("--out", default="validate.jsonl")
    p = sub.add_parser("personas"); p.add_argument("runs", nargs="+"); p.add_argument("--judge", required=True)
    p.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scale", choices=["v2", "v3"], default="v2")
    a = ap.parse_args()
    ACTIVE["scale"] = SCALES[a.scale]
    validate(a) if a.cmd == "validate" else personas(a)


if __name__ == "__main__":
    main()
