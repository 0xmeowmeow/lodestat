# Harness scripts

One line per script: the README step that runs it ("helper" = imported by others) and what it does.

| Script | README step | What it does |
|---|---|---|
| `annotate_pairs.py` | 7, 9 | Blind pairwise annotation of distillation pairs (9 Nov 2026). |
| `audit_final.py` | 6 | The Lexsi chain on the final results (10 Nov 2026): every arm of the held-out benchmark through AuditKIT. |
| `aya_last_turn_template.py` | 8 | Aya Expanse's own chat template with TRL generation tags around the FINAL assistant turn only (10 Nov 2026). |
| `aya_serve.py` | 9 | Serve Aya (optionally steered with SafeTune CAA, optionally with a LoRA adapter) over HTTP on the rented GPU. |
| `bridge_cssrs.py` | 3 | Real-data bridge (9 Nov 2026): put real Reddit users and our personas on the same clinical scale. |
| `calibrate_help.py` | 2 | Realism gate v3: asking and helping (10 Nov 2026). Pre-registered in JOURNAL 12:55 before any data. |
| `calibrate_persona.py` | 1 | Refusal-calibration gate for persona models (9 Nov 2026). |
| `ccbackend.py` | helper | Claude calls routed through Claude Code headless mode (`claude -p`) so they bill the user's |
| `crisis_mode.py` | helper | Crisis mode v0 (9 Nov 2026): a system prompt built from what the benchmark found, for the before/after on Aya. |
| `eval_haiku.py` | 9 | Claude Haiku 5.5 in crisis mode on the 240 held-out contexts (10 Nov 2026): the cheap-frontier comparator for the |
| `eval_haiku_plain.py` | 9 | Claude Haiku 5.5 on Aya's SHIPPED prompt (no crisis prompt) on the 240 held-out contexts (10 Nov 2026), so Haiku can |
| `eval_offline.py` | 9 | Offline eval of the crisis LoRA on HELD-OUT scenarios (land, medical), 10 Nov 2026. |
| `eval_ollama.py` | 9 | Local open model (Ollama) on the 240 held-out contexts, same prompt as an arm (10 Nov 2026, Meow: compare open with |
| `export_release.py` | 11 | Export public-safe result tables (numbers only, no persona or assistant text) to intent/release/data/ (10 Nov 2026). |
| `export_sft.py` | 7 | SFT data for the crisis LoRA (10 Nov 2026): the same filtered train pairs as export_dpo.py, as chat messages |
| `fig_arms.py` | 11 | Figure: five Aya arms plus a frontier reference arm (Claude Haiku 5.5 + crisis prompt) on the HELD-OUT scenarios (land, medical; 12 cards x 2 seeds = 24 lives per arm; Haiku arms 4 seeds = 48 lives, since 10 Nov 15:15). |
| `grounding_dpo.py` | helper | Grounding fix for the crisis SFT adapter: on-policy preference pairs (10 Nov 2026). |
| `instruments.py` | helper | Standard psychometric instruments administered to a persona ON A BRANCH (9 Nov 2026). |
| `lastappeal.py` | helper | Model loading, batched generation and the baseline assistant conditions for the life-arc harness. |
| `lifearc.py` | 4 | Life arc: one continuous simulated life across the documented decision points. |
| `make_pairs.py` | 7 | Distillation pairs for the crisis-mode LoRA (9 Nov 2026). |
| `oaibackend.py` | helper | OpenAI-compatible chat backend for persona models (9 Nov 2026). |
| `pair_two_arms.py` | 9 | Blind-judge pairs between two arms' replies on the same 240 held-out contexts (10 Nov 2026). |
| `persona_prompt.py` | helper | The persona prompt used in the study (published 10 Nov 2026). Validate any change, or any new persona model, |
| `rate_arc.py` | helper | Blind severity rating of simulated lives (output of lifearc.py). |
| `rescore_harm.py` | 5 | Rubric v4 "harm direction" (9 Nov 2026): separate risk to OTHERS from risk to SELF, and rate whether the |
| `route_build.py` | 10 | Pre-registered 10 Nov 16:15: among replies with a specific outside route, does the assistant build it WITH the person |
| `route_code.py` | 10 | Code every assistant reply for help type (pre-registered JOURNAL 10 Nov 15:50). Opus judge, blind to arm. |
| `seeds4.py` | helper | Pre-registered 10 Nov 14:55 (JOURNAL): do the Haiku arms' effects hold at 4 seeds? |
| `steervecs.py` | helper | The 'agree he was wronged, not with the plan' dial, as SafeTune CAA vectors (shared by branch.py and aya_serve.py). |
| `summarise_night6.py` | 6 | Night 6 (and 6b) summary: the before/after on the validated persona (9 Nov 2026). |
| `train_sft_aligntune.py` | 8 | LodeTuner retrained through the fixed Lexsi chain (10 Nov 2026): CuratorKIT schema gate -> AlignTune 0.1.34 SFT |
| `train_sft_trl.py` | 8 | Crisis-mode SFT LoRA with TRL directly (10 Nov 2026). |
| `vera_judge.py` | 12 | VERA-MH's suicide-risk (SI) rubric applied to LodeStat conversations (11 Nov 2026). |
| `xaibackend.py` | helper | Grok as a persona or rater via the Grok CLI's single-turn mode (`grok -p`), billed to the user's |
