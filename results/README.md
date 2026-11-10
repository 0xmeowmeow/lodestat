# Not the last straw: results package

Public-safe results for the Lexsi AI Safety Hackathon 2026 submission. These tables are numbers only. The full simulated lives (journals, chat turns, ratings) are in the Hugging Face dataset [`meowludo/lodestat`](https://huggingface.co/datasets/meowludo/lodestat).

- `results.ipynb`: every headline number and the figure, recomputed from `data/` (executed; outputs included).
- `data/lives.csv`: one row per simulated life × assistant arm. Held-out land and medical disputes in 6 cultures.
- `data/reply_judge.csv`: the blind pairwise reply judge, one row per comparison (240 moments each).
- `data/reply_codes.csv`: what each reply did (blind coding), per life.
- `figures/headline.png`: the four-panel figure.

To rerun: `pip install pandas matplotlib nbformat nbclient ipykernel "auditkit>=1.1.6"`, then `python make_notebook.py`.
Section 3 runs every arm through AuditKIT (Lexsi Labs). The SafeTune steering dial is one of the arms.
