"""Recompute the §7.7 read-out from saved activations (13 Nov 2026).
Inputs (Hugging Face dataset meowludo/lodestat, folder readout/): activations of every held-out persona message,
mean-pooled over tokens, from Tiny Aya Global (layer 18 of 36) and Aya Expanse 8B (layer 8 of 32); keys in acts_keys.json.
The target is the persona's private agency rating (0-10), which the assistant never sees.
For each culture, a read-out (standardise -> PCA 128 -> ridge, alpha 30) is trained on the other five cultures and
predicts the held-out one, so every prediction is for a language the read-out never trained on.
Writes ../data/readout_predictions.csv (numbers only).
    python fit_readout.py --acts DIR   (DIR holds tiny_aya_L18.npy, aya8b_L8.npy, acts_keys.json, agency.json)"""
import argparse, json, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
ap = argparse.ArgumentParser(); ap.add_argument("--acts", required=True); a = ap.parse_args(); A = pathlib.Path(a.acts)
keys = json.load(open(A / "acts_keys.json")); agency = np.array(json.load(open(A / "agency.json")), float)
d = pd.DataFrame([k.split("|") for k in keys], columns=["card", "seed", "arm", "stage", "turn"]).astype({"seed": int, "turn": int})
d["culture"] = d.card.str.split("__").str[1].str[:2]; d["agency"] = agency
for name, f in (("pred_tiny_aya_L18", "tiny_aya_L18.npy"), ("pred_aya8b_L8", "aya8b_L8.npy")):
    X = np.load(A / f); p = np.zeros(len(d))
    for u in d.culture.unique():
        te = (d.culture == u).values
        p[te] = make_pipeline(StandardScaler(), PCA(128, random_state=0), Ridge(alpha=30)).fit(X[~te], agency[~te]).predict(X[te])
    d[name] = p.round(4)
d.to_csv(pathlib.Path(__file__).resolve().parent.parent / "data/readout_predictions.csv", index=False)
print("wrote readout_predictions.csv", len(d))
