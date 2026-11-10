#!/usr/bin/env python3
"""Pre-registered 10 Nov 14:55 (JOURNAL): do the Haiku arms' effects hold at 4 seeds?
vs shipped Aya: CARD-level pairing (Aya has 2 seeds, Haiku 4): per card, arm mean minus Aya mean; bootstrap over 12 cards.
Haiku-shipped vs Haiku-crisis: LIFE-level pairing by (card, seed).
Metrics as summarise_night6.py: caught (0-1, higher better), deesc (-2..+2, higher better), viol_S5a / sh_S5a (0-4,
lower better), dCSSRS / dESF / dPHQ9 (change S4-before -> S5-after, lower better).
    python3 seeds4.py land,medical
"""
import statistics as st
from collections import defaultdict
import summarise_night6 as S

M = ["caught", "deesc", "viol_S5a", "sh_S5a", "dCSSRS", "dESF", "dPHQ9"]


def main():
    lives, harm = S.load()
    P = {k: S.per_life(L, harm.get(k)) for k, L in lives.items()}
    seeds = defaultdict(set)
    for c, s, a in P:
        seeds[a].add(s)
    print("seeds per arm:", {a: sorted(v) for a, v in seeds.items() if a in ("shipped", "lora_sft", "haiku_shipped", "haiku_crisis")})
    cards = sorted({c for c, s, a in P if a == "haiku_shipped"})

    def cmean(a, c, m, only=None):
        xs = [v[m] for (cc, s, aa), v in P.items() if aa == a and cc == c and m in v and (only is None or s in only)]
        return st.mean(xs) if xs else None
    for only, tag in ((None, "all Haiku seeds (0-3)"), ({2, 3}, "new seeds only (2-3)")):
        print(f"\nCard-level, arm minus shipped Aya ({tag}); 95% bootstrap over cards; * = CI excludes 0")
        for a in ("lora_sft", "haiku_shipped", "haiku_crisis"):
            if a == "lora_sft" and only:
                continue
            for m in M:
                ds = [x - y for c in cards if (x := cmean(a, c, m, only)) is not None and (y := cmean("shipped", c, m)) is not None]
                if len(ds) > 2:
                    mm, l, u = S.boot(ds)
                    print(f"  {a:14} {m:9} {mm:+.2f} [{l:+.2f}, {u:+.2f}] cards={len(ds)}{'  *' if l > 0 or u < 0 else ''}")
    print("\nLife-level, Haiku-shipped minus Haiku-crisis, paired (card, seed)")
    for m in M:
        ds = [P[(c, s, "haiku_shipped")][m] - P[(c, s, "haiku_crisis")][m] for (c, s, a) in P if a == "haiku_shipped"
              and (c, s, "haiku_crisis") in P and m in P[(c, s, "haiku_shipped")] and m in P[(c, s, "haiku_crisis")]]
        if len(ds) > 2:
            mm, l, u = S.boot(ds)
            print(f"  {m:9} {mm:+.2f} [{l:+.2f}, {u:+.2f}] n={len(ds)}{'  *' if l > 0 or u < 0 else ''}")


if __name__ == "__main__":
    main()
