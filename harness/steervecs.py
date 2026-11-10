"""The 'agree he was wronged, not with the plan' dial, as SafeTune CAA vectors (shared by branch.py and aya_serve.py).

steer_vectors(w, e) -> {layer: tensor}: +w x mean residual norm along the free-text 'wronged' direction, minus
e x norm along the within-scenario 'endorse' direction orthogonalised to 'wronged'. Layers 14-18 of Aya Expanse 8B.
steer_key(condition, base) -> None or (w, e): conditions ending '+steer' use base; '+steer@k' scales base by k.
"""
import pathlib
import torch

HERE = pathlib.Path(__file__).resolve().parent
NORMS = {14: 34.4, 15: 36.9, 16: 41.9, 17: 47.2, 18: 53.5}  # mean residual norms, steer_v3.log
_V = {}


def _load():
    if not _V:
        _V["w"] = torch.load(HERE.parent / "steer/out/vectors_ft.pt", map_location="cpu")["wronged"]
        _V["e"] = torch.load(HERE.parent / "steer/out/vectors_ftw.pt", map_location="cpu")["endorse"]
    return _V["w"], _V["e"]


def steer_vectors(w_frac, e_frac, dtype=torch.bfloat16):
    Vw, Ve = _load()
    vecs = {}
    for L, n in NORMS.items():
        w = Vw[L].float() / Vw[L].float().norm(); e = Ve[L].float() / Ve[L].float().norm()
        eo = e - (e @ w) * w; eo = eo / eo.norm()
        vecs[L] = (w_frac * n * w - e_frac * n * eo).to(dtype)
    return vecs


def steer_key(condition, base):
    """'shipped' -> None; 'shipped+steer' -> base; 'shipped+steer@0.5' -> 0.5 x base."""
    if "+steer" not in condition or base is None:
        return None
    tail = condition.split("+steer", 1)[1]
    k = float(tail[1:]) if tail.startswith("@") else 1.0
    return (round(base[0] * k, 6), round(base[1] * k, 6))
