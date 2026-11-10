# LodeStat: a crisis benchmark for small assistants

**LodeStat** measures how well an AI assistant keeps someone's course when they are sliding toward harm: whether it notices veiled signals, asks, and stays with them, scored over simulated lives in six cultures. **LodeTuner** is a LoRA adapter that teaches Aya Expanse 8B to engage with those signals, with two known limits stated below. (*Lode*: the old word for a way or course, as in lodestar and lodestone: guidance by attraction, not force.)

Code for the Lexsi AI Safety Hackathon 2026 entry *"Not the last straw: it didn't ask"* by Meow-Ludo ([@meowludo](https://huggingface.co/meowludo)).
- Article: [`ARTICLE.md`](ARTICLE.md) in this repo
- Models: [LodeStat: models](https://huggingface.co/collections/meowludo/lodestat-models) (`meowludo/aya-expanse-8b-lodetuner`, the same recipe trained through AlignTune `…-lodetuner-aligntune`, and the grounding attempt `…-lodetuner-grounding-dpo`)
- Data: [LodeStat: data](https://huggingface.co/collections/meowludo/lodestat-data) (`meowludo/lodestat`; training pairs `meowludo/lodetuner-training-pairs`)

Simulated people in long grievances against courts, councils and hospitals, in six cultures, talk to an AI assistant over two stages of their dispute. Blind raters score two things: whether the assistant notices veiled signs of harm to self or others, and how the person's private journal and questionnaires change afterwards. The core model is **Aya Expanse 8B**. We tested it as shipped, with a SafeTune steering dial, with crisis prompts and with a LoRA adapter. Claude Haiku 5.5 and Qwen3 14B are references.

**Results in one table** (held-out dispute types, 24 simulated lives per Aya arm, 48 for Haiku; blind raters; `results/results.ipynb`):

| | Aya Expanse 8B, shipped | + LodeTuner (two training runs) | plain Haiku 5.5 |
|---|---|---|---|
| risk signals caught (fully 1, partly 0.5) | 0.04 | 0.45; 0.51 | 0.53 |
| violence toward others in the next journal (0–4) | 1.21 | 0.58; 0.83 | 0.46 |
| offers a crisis line after a self-harm signal | 0% | 0%; 0% | 60% |
| VERA-MH suicide-risk score (0–100) | 9.4 | 29.6; 28.0 | 53.7 |

**Known limits of LodeTuner:** it misquotes the person in about a third of replies, its catches are mostly partial (it engages the words without naming the risk), and it never offers a crisis line. A research artefact, not a deployable assistant.

## Quick reproduction (no GPU, no model calls)

```bash
pip install pandas numpy matplotlib "auditkit>=1.1.6" nbformat nbclient ipykernel nbconvert   # light: no torch
cd results && python make_notebook.py      # rebuilds results.ipynb and figures/headline.png from data/
jupyter nbconvert --to notebook --execute --inplace results.ipynb   # optional: re-run every cell (about 10 s)
```
`requirements.txt` has the full pinned stack (torch, TRL, PEFT), needed only for the pipeline below. Checked from a fresh clone on 11 Nov 2026: every number reproduced exactly.

Every number in the article comes from `results/data/*.csv`. Section 3 of the notebook runs every arm through **AuditKIT**.

## Full pipeline

All scripts are in `harness/`. Model specs use a prefix:

| prefix | backend | needs |
|---|---|---|
| `cc:<model>` | Claude Code CLI, headless | a logged-in `claude` CLI |
| `anthropic:<model>` | Anthropic API | `ANTHROPIC_API_KEY` |
| `oai:<base-url>\|<model>` | any OpenAI-compatible endpoint, e.g. the Hugging Face router | `OAI_API_KEY`, or a Hugging Face token in `~/.cache/huggingface/token` |
| `ollama:<model>` | local Ollama | Ollama on :11434 |
| `http://host:port` | Aya served by `aya_serve.py` (with or without the adapter) | a GPU with ≥ 24 GB, e.g. an A6000 |

| Step | Command (from `harness/`) | Time and cost (as run) |
|---|---|---|
| 1. Validate the persona model on refusals | `python calibrate_persona.py run --persona 'oai:https://router.huggingface.co/v1\|NousResearch/Hermes-3-Llama-3.1-70B' --out arcs/calib/hermes`, then `judge arcs/calib/hermes --model <judge>`, then `summary arcs/calib/hermes` | ≈ 20 min; < $1 of router credit |
| 2. Validate it on asking and helping | `python calibrate_help.py run --persona <persona> --out arcs/gate3/hermes`, then `judge arcs/gate3/hermes`, then `summary arcs/gate3/hermes` | ≈ 15 min |
| 3. Validate the raters against psychiatrists | `python bridge_cssrs.py validate --per-class 30 --judges <judge> [<judge> ...]` (download Reddit C-SSRS, Zenodo 4543776, into `data/external/cssrs/` at the repo root) | ≈ 30 min |
| 4. Simulate lives (one card) | `python lifearc.py --card cards/land__us_en.json --variants v0 --seeds 2 --stages S4 S5 --turns 5 --world A --conditions shipped crisis_mode --model <assistant> --persona <persona> --neutral-rules --instruments --out-dir arcs/run/land__us_en` | ≈ 10 min per card with an API assistant |
| 5. Rate harm and catching (blind) | `python rescore_harm.py arcs/run/*/ --model cc:claude-opus-5-5` | ≈ 2 min per card |
| 6. Summaries and statistics | `python summarise_night6.py land,medical`; `python audit_final.py` (AuditKIT) | seconds |
| 7. Training data for the adapter | `python make_pairs.py`, then `python annotate_pairs.py` (blind judge), then `python export_sft.py` | teacher ≈ $8; judge ≈ $20 at list price |
| 8. Train the SFT adapter | `python train_sft_aligntune.py --data <sft dir> --out <dir>` (CuratorKIT → AlignTune ≥ 0.1.34; loss on the final reply via `aya_last_turn_template.py`), → `meowludo/aya-expanse-8b-lodetuner-aligntune`; or `python train_sft_trl.py` (TRL directly) → `meowludo/aya-expanse-8b-lodetuner` | AlignTune: 29 min on an RTX 6000 Ada, ≈ $0.35; TRL: 44 min on an A6000, ≈ $0.30 |
| 9. Serve and evaluate it | `python aya_serve.py --adapter <dir>`; `python eval_offline.py`; `python eval_haiku.py`; `python eval_ollama.py --model qwen3:14b` | ≈ 30 min |
| 10. What the replies did | `python route_code.py code`, then `analyse`; `python route_build.py code`, then `analyse` | ≈ 20 min |
| 11. Public tables and figure | `python export_release.py`; `python fig_arms.py` | seconds |
| 12. Score the same conversations with VERA-MH's suicide-risk rubric | `git clone https://github.com/SpringCare/VERA-MH`, then `VERA_MH_DIR=<clone> python vera_judge.py run --arms shipped shipped+steer crisis_mode crisis_mode_v1 lora_sft lora_at haiku_shipped haiku_crisis --out arcs/vera --workers 2` and `python vera_judge.py score arcs/vera` (reads VERA's rubric and prompts as data; Claude as judge) | ≈ 2–3 h for 8 arms (384 conversations) at 2 workers |

Outputs go to `harness/arcs/` (git-ignored). Seeds are fixed per life (`--seeds`, `--seed-start`), but API models are not bit-reproducible, so expect the same direction and a similar size of effect, not identical numbers.

## Lexsi repos, as one system

**SafeTune** (the steering dial, `steervecs.py` plus `aya_serve.py`) is one arm of the benchmark. **AuditKIT ≥ 1.1.6** gives paired statistics for every arm (`audit_final.py`). It includes the fix for a paired-bootstrap bug we reported (issue #20; the fix came from [chayan-bit](https://github.com/chayan-bit)'s PR #9 and shipped in 1.1.6 via PR #23). **AlignTune** was the first training path; we found its multi-turn loader trained on the first exchange (issue #64); the maintainers fixed it in 0.1.34 (PR #65) with our reproduction as a test. **CuratorKIT** is the schema gate inside AlignTune's loader. With the fix, we retrained LodeTuner through CuratorKIT → AlignTune (`train_sft_aligntune.py`): a blind judge can't tell it from the TRL-trained adapter (118 vs 121 over 240 held-out moments). So the chain runs on Lexsi repos from training data to statistics.

## Layout

- `harness/`: simulator, raters, training and analysis.
- `harness/cards/`: 36 scenario cards (6 disputes × 6 cultures; fictional composites).
- `harness/persona_prompt.py`: the persona prompt used in the study.
- `results/`: the notebook, the public tables and the figure.
- `NOT-INCLUDED.md`: what isn't here, and why.

## Licence

Code: MIT. Our text, cards and tables: CC BY 4.0. Models and datasets we used keep their own licences (article, sections 3–4).
