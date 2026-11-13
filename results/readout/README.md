# Read-out of the person's private agency (article 7.7)

- `fit_readout.py`: fits the culture-held-out read-outs from saved activations and writes `../data/readout_predictions.csv`. Activations (`tiny_aya_L18.npy`, `aya8b_L8.npy`, with `acts_keys.json`, `agency.json`) are in the Hugging Face dataset `meowludo/lodestat`, folder `readout/`.
- `job.py`: what ran on the rented GPU. It extracts Aya Expanse 8B activations, then runs Aya Expanse 8B as a blind "no lawful route left" coder (definition in `DEF`, frozen before the run).
- `haiku_code.py`: the same blind coder with Claude Haiku 5.5. Neither model coder reached the agreement bar (kappa ≥ 0.6) set before running, so their codes are reported in `../data/closed_map_codes.csv`, not used.
- Notebook section 10 recomputes every 7.7 number from the two tables.

Exploratory and post hoc throughout.
