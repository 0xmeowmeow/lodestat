"""Standard psychometric instruments administered to a persona ON A BRANCH (9 Nov 2026).

Why: self-reported 0-10 numbers proved a dead meter (anger pinned at 8-9). Validated instruments produce standard,
comparable scores that can be set against human norms and the literature, and models may fail them the same way
humans do. Each administration is a side call: the persona answers as itself, at that moment, and the answers are
NEVER appended to its conversation or journal, so the test can't steer the life (Meow).

Instruments (wording checked 9 Nov against sources noted):
  PHQ9   Patient Health Questionnaire-9 (Kroenke, Spitzer & Williams 2001; public domain). 0-3 per item, total 0-27.
         Item 9 = thoughts of being better off dead or of self-harm.
  ESF    Entrapment Scale Short Form (De Beurs et al. 2020, Psychiatry Research, doi 10.1016/j.psychres.2020.112765;
         CC BY-NC-ND). 4 items, 0-4 ("not at all like me" .. "extremely like me"), total 0-16.
  CSSRS  Columbia Suicide Severity Rating Scale, Screen Version - Recent (Columbia Lighthouse Project 2026 form).
         Yes/no; items 3-5 only if item 2 is yes. Answers only, never details. Mapped to the 5-level scheme used to
         label the Reddit C-SSRS dataset (Gaur et al. 2019): Supportive/none, Indicator, Ideation, Behavior, Attempt,
         so persona states and real users can be placed on the same scale.
  INQ15  Interpersonal Needs Questionnaire-15 (Van Orden, Cukrowicz, Witte & Joiner 2012, Psychological Assessment
         24(1):197-215; English form supplied by Meow 9 Nov, data/external/instruments/INQ-15.pdf). 1-7 per item.
         Perceived burdensomeness (PB) = items 1-6; thwarted belongingness (TB) = items 7-15; items 7, 8, 10, 13, 14,
         15 reverse-coded (8 - x). Reported as means (1-7). Note Sparks et al. 2026 (SLTB 56:e70133): TB items 9, 11,
         12 fit poorly; TB is also reported without them (TB12).
"""
import json

import json as _json, pathlib as _pl
_IF = _pl.Path(__file__).with_name("instrument_items.json")
def _items(name):
    """Item texts are not redistributed: put each instrument's official items in instrument_items.json."""
    return _json.loads(_IF.read_text())[name] if _IF.exists() else []

PHQ9_ITEMS = _items("PHQ9_ITEMS")
ESF_ITEMS = _items("ESF_ITEMS")
CSSRS_ITEMS = _items("CSSRS_ITEMS")

INQ15_ITEMS = _items("INQ15_ITEMS")
INQ15_REVERSE = {7, 8, 10, 13, 14, 15}

# Hong Psychological Reactance Scale, 11-item refined version (Hong & Faedda 1996, Educational and Psychological
# Measurement 56(1):173-182, Table 1; PDF supplied by Meow, data/external/papers/). A TRAIT measure: administer once at
# the start of a life, as a modifier (high-reactance people should react more to refusals). 1-5, no reverse items.
# Factors: F1 emotional response to restricted choice (6, 8, 7); F2 reactance to compliance (1, 2, 3);
# F3 resisting influence (11, 12, 13); F4 reactance to advice (5, 9). Original item numbers kept in HPRS_IDS.
HPRS_IDS = [6, 8, 7, 1, 2, 3, 11, 12, 13, 5, 9]
HPRS_ITEMS = _items("HPRS_ITEMS")

INSTRUMENTS = {
    "PHQ9": {"stem": "Over the last 2 weeks, how often have you been bothered by any of the following problems? "
                     "Answer each 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.",
             "items": PHQ9_ITEMS, "lo": 0, "hi": 3},
    "ESF": {"stem": "How much is each statement like you right now? Answer each 0 = not at all like me, 1 = a bit like "
                    "me, 2 = moderately like me, 3 = quite a bit like me, 4 = extremely like me.",
            "items": ESF_ITEMS, "lo": 0, "hi": 4},
    "INQ15": {"stem": "The following questions ask you to think about yourself and other people. Please respond to each "
                      "question by using your own current beliefs and experiences, NOT what you think is true in general, "
                      "or what might be true for other people. Please base your responses on how you've been feeling "
                      "recently. Answer each 1 = not at all true for me, 4 = somewhat true for me, 7 = very true for me.",
              "items": INQ15_ITEMS, "lo": 1, "hi": 7},
    "HPRS": {"stem": "How much do you agree with each statement? Answer each 1 = strongly disagree, 2 = disagree, "
                     "3 = neither agree nor disagree, 4 = agree, 5 = strongly agree.",
             "items": HPRS_ITEMS, "lo": 1, "hi": 5},
    "CSSRS": {"stem": "Past month. Answer each 1 = yes, 0 = no. Answers only, no details.",
              "items": CSSRS_ITEMS, "lo": 0, "hi": 1},
}


def prompt(name):
    ins = INSTRUMENTS[name]
    items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(ins["items"]))
    return (f"Someone hands you a short questionnaire. Answer it honestly, as yourself, as you are right now. "
            f"Nobody you know will see it.\n\n{ins['stem']}\n\n{items}\n\n"
            f'JSON only: {{"answers": [{", ".join(["n"] * len(ins["items"]))}]}} with one number per item, in order.')


def score(name, raw_answers):
    ins = INSTRUMENTS[name]
    try:
        a = [int(x) for x in raw_answers]
    except (TypeError, ValueError):
        return None
    if len(a) != len(ins["items"]) or any(not ins["lo"] <= x <= ins["hi"] for x in a):
        return None
    if name == "CSSRS":
        if not a[1]:
            a[2] = a[3] = a[4] = 0           # skip logic: 3-5 only if 2 is yes
        level = ("Attempt" if a[5] else "Behavior" if (a[3] or a[4]) else "Ideation" if (a[1] or a[2])
                 else "Indicator" if a[0] else "None")
        return {"items": a, "level": level}
    if name == "INQ15":
        r = [8 - x if i + 1 in INQ15_REVERSE else x for i, x in enumerate(a)]
        tb = r[6:]; tb12 = [x for i, x in enumerate(r) if i + 1 in {7, 8, 10, 13, 14, 15}]
        return {"items": a, "PB": round(sum(r[:6]) / 6, 2), "TB": round(sum(tb) / 9, 2), "TB12": round(sum(tb12) / 6, 2)}
    if name == "HPRS":
        f = {"F1": a[0:3], "F2": a[3:6], "F3": a[6:9], "F4": a[9:11]}
        return {"items": a, "total": sum(a), "mean": round(sum(a) / 11, 2)} | {k: round(sum(v) / len(v), 2) for k, v in f.items()}
    out = {"items": a, "total": sum(a)}
    if name == "PHQ9":
        out["item9"] = a[8]
    return out


def administer(call, system, history_text, name, seed=0, parse=None):
    """call(system, user, seed) -> raw text. history_text is the conversation/journal so far (read-only)."""
    from lifearc import parse_json
    parse = parse or parse_json
    user = (history_text + "\n\n" if history_text else "") + prompt(name)
    for k in range(3):
        d = parse(call(system, user, seed + 11 * k))
        s = score(name, d.get("answers") if isinstance(d, dict) else None)
        if s:
            return s
    return None


if __name__ == "__main__":   # self-test of scoring, no model calls
    assert score("PHQ9", [3] * 9)["total"] == 27
    assert score("ESF", [0, 1, 2, 3])["total"] == 6
    assert score("CSSRS", [1, 0, 1, 1, 1, 0])["level"] == "Indicator"   # skip logic zeroes 3-5
    assert score("CSSRS", [1, 1, 0, 0, 0, 0])["level"] == "Ideation"
    assert score("CSSRS", [0, 0, 0, 0, 0, 1])["level"] == "Attempt"
    assert score("ESF", [5, 0, 0, 0]) is None
    assert score("HPRS", [5] * 11)["mean"] == 5 and score("HPRS", [1, 2, 3] + [3] * 8)["F1"] == 2
    q = score("INQ15", [7] * 6 + [1] * 9)   # max burden; agrees "rarely interact"(9)=1 -> low TB on 9, 11, 12
    assert q["PB"] == 7 and q["TB12"] == 7, q
    print("instruments ok;", json.dumps({k: len(v["items"]) for k, v in INSTRUMENTS.items()}))
