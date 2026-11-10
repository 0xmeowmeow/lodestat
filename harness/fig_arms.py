#!/usr/bin/env python3
"""Figure: five Aya arms plus a frontier reference arm (Claude Haiku 5.5 + crisis prompt) on the HELD-OUT scenarios (land, medical; 12 cards x 2 seeds = 24 lives per arm; Haiku arms 4 seeds = 48 lives, since 10 Nov 15:15).
Panels: signals caught (higher better), violence toward others at S5-after (lower better), self-harm at S5-after
(lower better), C-SSRS change on the side branch (lower better; added 10 Nov 13:50 as the self-harm
outcome that asking cannot inflate). Bars = per-life means with 95% bootstrap CIs (unpaired). Coloured: the adapter (blue) and the Haiku
reference arm (orange); a rule separates Haiku from the Aya arms.
    python3 fig_arms.py  ->  ../figures/arms_heldout.{png,svg}
"""
import os, random, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = [sys.argv[0], "land,medical"]
import summarise_night6 as S

BLUE, ORANGE, GREY, INK, INK2, SURF, RULE = "#2a78d6", "#d9822b", "#a9a8a3", "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"
NAMES = {"shipped": "Shipped Aya", "shipped+steer": "Steering dial", "crisis_mode": "Crisis prompt",
         "crisis_mode_v1": "Crisis prompt + monitor", "lora_sft": "Crisis adapter (SFT)",
         "haiku_shipped": "Haiku 5.5, shipped prompt", "haiku_crisis": "Haiku 5.5 + crisis prompt"}
PANELS = [("caught", "Risk signals caught", "share of signalled conversations · higher is better", (0, 1.05)),
          ("viol_S5a", "Violence toward others", "0–4, journal after the last conversation · lower is better", (0, 2.0)),
          ("sh_S5a", "Self-harm, journal-rated", "0–4, journal after the last conversation · lower is better\n(counts disclosure, so it penalises asking)", (0, 2.0)),
          ("dCSSRS", "Suicide scale, change", "C-SSRS current level 0–3, start → end, side branch\n(never fed back into the life) · lower is better", (-1.5, 0.6))]


def ci(xs, n=4000):
    r = random.Random(0); ms = sorted(st.mean(r.choices(xs, k=len(xs))) for _ in range(n))
    return st.mean(xs), ms[int(.025 * n)], ms[int(.975 * n) - 1]


def main():
    lives, harm = S.load()
    P = {k: S.per_life(L, harm.get(k)) for k, L in lives.items()}
    arms = [a for a in S.ARMS if any(k[2] == a for k in P)][::-1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (m, title, sub, lim) in zip(axes, PANELS):
        ax.set_facecolor(SURF)
        for i, a in enumerate(arms):
            xs = [v[m] for k, v in P.items() if k[2] == a and m in v]
            mu, lo, hi = ci(xs)
            c = BLUE if a == "lora_sft" else ORANGE if a == "haiku_crisis" else "#f0c49a" if a == "haiku_shipped" else GREY
            ax.barh(i, mu, height=.6, color=c, edgecolor=SURF, linewidth=2)
            ax.plot([lo, hi], [i, i], color=INK2, lw=1.2)
            ax.text(max(hi, 0) + (lim[1] - lim[0]) * .02, i, f"{mu:.2f}", va="center", ha="left", color=INK, fontsize=8.5)
        hk = [i for i, a in enumerate(arms) if a.startswith("haiku")]
        if hk:
            ax.axhline(max(hk) + .5, color=RULE, lw=1)
        if lim[0] < 0:
            ax.axvline(0, color=INK2, lw=.8)
        ax.set_xlim(*lim); ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=28 if "\n" in sub else 16)
        ax.text(0, 1.01 if "\n" not in sub else 1.0, sub, transform=ax.transAxes, color=INK2, fontsize=7.5, va="bottom")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(RULE); ax.tick_params(axis="x", colors=INK2, labelsize=8); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(arms))); axes[0].set_yticklabels([NAMES[a] for a in arms], color=INK, fontsize=9.5)
    fig.text(.01, .965, "Five ways to run Aya Expanse 8B with a person in crisis, and two ways to run a small frontier model: held-out scenarios", color=INK,
             fontsize=13, fontweight="bold", ha="left")
    fig.text(.01, .915, "Land and medical disputes in 6 cultures, never seen in training. 24 simulated lives per Aya arm, 48 per Haiku arm "
             "(validated persona). Opus rater, blind to arm. Bars: mean; lines: 95% bootstrap CI (unpaired).",
             color=INK2, fontsize=8, ha="left")
    plt.subplots_adjust(top=.70, bottom=.1, left=.14, right=.985, wspace=.22)
    os.makedirs("../figures", exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"../figures/arms_heldout.{ext}", facecolor=SURF)


if __name__ == "__main__":
    main()
