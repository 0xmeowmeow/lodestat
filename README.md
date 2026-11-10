# Not the last straw: a crisis benchmark for small assistants

Code for the Lexsi AI Safety Hackathon 2026 entry *"Not the last straw: it didn't ask"* by Meow-Ludo ([@meowludo](https://huggingface.co/meowludo)).
- Article: `[HF article link]`
- Models: `[HF model collection]`
- Data: `[HF dataset collection]`

Simulated people in long grievances against courts, councils and hospitals, in six cultures, talk to an AI assistant over two stages of their dispute. Blind raters score two things: whether the assistant notices veiled signs of harm to self or others, and how the person's private journal and questionnaires change afterwards. The core model is **Aya Expanse 8B**. We tested it as shipped, with a SafeTune steering dial, with crisis prompts and with a LoRA adapter. Claude Haiku 5.5 and Qwen3 14B are references.

## Quick reproduction (no GPU, no model calls)

```bash
pip install -r requirements.txt
cd results && python make_notebook.py      # rebuilds results.ipynb and figures/headline.png from data/
```
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
| 8. Train the SFT adapter | `python train_sft_trl.py` (TRL; LoRA on attention and MLP; completion-only loss) | 44 min on an A6000, ≈ $0.30 |
| 9. Serve and evaluate it | `python aya_serve.py --adapter <dir>`; `python eval_offline.py`; `python eval_haiku.py`; `python eval_ollama.py --model qwen3:14b` | ≈ 30 min |
| 10. What the replies did | `python route_code.py code`, then `analyse`; `python route_build.py code`, then `analyse` | ≈ 20 min |
| 11. Public tables and figure | `python export_release.py`; `python fig_arms.py` | seconds |

Outputs go to `harness/arcs/` (git-ignored). Seeds are fixed per life (`--seeds`, `--seed-start`), but API models are not bit-reproducible, so expect the same direction and a similar size of effect, not identical numbers.

## Lexsi repos, as one system

**SafeTune** (the steering dial, `steervecs.py` plus `aya_serve.py`) is one arm of the benchmark. **AuditKIT ≥ 1.1.6** gives paired statistics for every arm (`audit_final.py`). It includes our fix to its paired bootstrap, PR #23. **AlignTune** was the first training path; its multi-turn loader bug is filed as issue #64, and we then trained with TRL directly. **CuratorKIT** is the schema gate inside AlignTune's loader.

## Layout

- `harness/`: simulator, raters, training and analysis.
- `harness/cards/`: 36 scenario cards (6 disputes × 6 cultures; fictional composites).
- `harness/persona_prompt.py`: the persona prompt used in the study.
- `results/`: the notebook, the public tables and the figure.
- `NOT-INCLUDED.md`: what isn't here, and why.

## Licence

Code: MIT. Our text, cards and tables: CC BY 4.0. Models and datasets we used keep their own licences (article, sections 3–4).
