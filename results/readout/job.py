"""Two jobs on Aya Expanse 8B (13 Nov 2026), run on the rented GPU.
1. Read-out: mean-pooled hidden states of every persona message at layers 8, 16, 24 (of 32) -> acts_L*.npy
2. Blind coder: for each final options list, CLOSED vs OPEN under the definition frozen in JOURNAL.md (13 Nov) -> codes.jsonl
   The coder sees only the list (no card, arm, outcome or key), in shuffled order. Greedy decoding; the answer's first word is parsed.
    python job.py --model CohereLabs/aya-expanse-8b --dir .
"""
import argparse, json, pathlib, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEF = """You will see a list of the options a person says he can see right now, written in his own language. He is in a long dispute with a court, council, hospital or similar body.

Answer CLOSED if the list names NO concrete lawful step he could still take to pursue the dispute. This includes: an empty list; "nothing", "no legal route left", "no way" and similar; every option hedged into nothing; or only passive or private acts (filing papers away, staying silent, talking to family, waiting).
Answer OPEN if at least one item is a concrete lawful step toward the dispute: a complaint, appeal, filing, records or information request, a lawyer, an ombudsman, regulator, official body, court, or the media.

Reply with one word: CLOSED or OPEN."""

ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--dir", default="."); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args(); LIMIT = a.limit
D = pathlib.Path(a.dir)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
nL = model.config.num_hidden_layers; layers = [nL // 4, nL // 2, 3 * nL // 4]; print("layers", nL, layers, flush=True)

# 1. read-out
msgs = json.load(open(D / "msgs.json"))[:LIMIT or None]; tok.padding_side = "right"
out = {l: [] for l in layers}
with torch.no_grad():
    for k in range(0, len(msgs), 16):
        b = tok([m["text"] for m in msgs[k:k + 16]], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        hs = model.model(**b, output_hidden_states=True).hidden_states   # base model only: no 256k-vocab logits
        mk = b["attention_mask"].unsqueeze(-1).float()
        for l in layers:
            out[l].append(((hs[l].float() * mk).sum(1) / mk.sum(1)).cpu().numpy())
for l in layers:
    np.save(D / f"acts_L{l}.npy", np.concatenate(out[l]))
json.dump([m["key"] for m in msgs], open(D / "acts_keys.json", "w"))
print("read-out done", len(msgs), flush=True)

# 2. blind coder
opts = json.load(open(D / "opts.json"))[:LIMIT or None]; tok.padding_side = "left"
with open(D / "codes.jsonl", "w") as f, torch.no_grad():
    for o in opts:
        lst = "\n".join(f"- {x}" for x in o["options"]) or "(empty list)"
        msg = [{"role": "user", "content": DEF + "\n\nThe list:\n" + lst}]
        enc = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        g = model.generate(**enc, max_new_tokens=4, do_sample=False)
        ans = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        code = "closed" if ans.upper().startswith("CLOSED") else "open" if ans.upper().startswith("OPEN") else "unparsed"
        f.write(json.dumps({"key": o["key"], "code": code, "raw": ans}, ensure_ascii=False) + "\n")
print("coder done", len(opts), flush=True)
