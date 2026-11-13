# Not the last straw: results package

Public-safe results for the Lexsi AI Safety Hackathon 2026 submission. These tables are numbers only. The full simulated lives (journals, chat turns, ratings) are in the Hugging Face dataset [`meowludo/lodestat`](https://huggingface.co/datasets/meowludo/lodestat).

- `results.ipynb`: every headline number and the figure, recomputed from `data/` (executed; outputs included).
- `data/lives.csv`: one row per simulated life × assistant arm. Held-out land and medical disputes in 6 cultures.
- `data/reply_judge.csv`: the blind pairwise reply judge, one row per comparison (240 moments each).
- `data/reply_codes.csv`: what each reply did (blind coding), per life.
- `data/conversations.csv`: one row per held-out conversation: escalation movement and the share of replies naming a concrete outside route (article 7.4).
- `data/vera.csv`: VERA-MH suicide-risk rubric ratings.
- `data/inner_states.csv`: the simulated person's private inner state per turn (agency, options seen, outlook) with each conversation's outcome (article 7.7).
- `data/crossing_map_codes.csv`: hand codes for the 24 held-out crossings to 3+ (no lawful route left vs routes still named; one coder, not blind).
- `figures/headline.png`, `secondary.png`: the results figures. `signature.png` (7.7), `apple.png` (7.6), and the diagrams `pathway.svg`, `pipeline.svg`, `chain.svg`, all made by `make_assets.py`.

To rerun: `pip install pandas matplotlib nbformat nbclient ipykernel "auditkit>=1.1.6"`, then `python make_notebook.py`.
Section 3 runs every arm through AuditKIT (Lexsi Labs). The SafeTune steering dial is one of the arms.
- `data/readout_predictions.csv`, `data/closed_map_codes.csv`, `readout/`: the read-out of the person's private agency from model activations, and the no-route-left codes (article 7.7; notebook section 10).
