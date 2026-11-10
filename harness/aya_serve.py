#!/usr/bin/env python3
"""Serve Aya (optionally steered with SafeTune CAA, optionally with a LoRA adapter) over HTTP on the rented GPU.

The persona and the rater (Claude, subscription) stay on the laptop; only the model under test runs here.
POST /gen {"convs": [[{role, content}, ...], ...], "max_new": 400, "batch_size": 16, "steer": [w, e] | null}
  -> {"replies": [...]}
GET /health -> {"ok": true, "model": ...}
Each request touches rig's keepalive file, so the idle watchdog only fires after real idleness.
    python aya_serve.py --model CohereLabs/aya-expanse-8b --port 8901 [--adapter DIR]
"""
import argparse, json, pathlib, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "vendor/SafeTune/src"))
try:
    from steervecs import steer_vectors  # noqa: E402   (SafeTune dial: not included, see PRIVATE_SLOTS.md)
except ImportError:
    steer_vectors = None
from safetune.interventions.steer.caa import CAAModel  # noqa: E402

KEEPALIVE = pathlib.Path("/workspace/state/keepalive")
LOCK = threading.Lock()


def keepalive():
    try:
        KEEPALIVE.parent.mkdir(parents=True, exist_ok=True); KEEPALIVE.touch()
    except OSError:
        pass  # not on a rig box
VEC_CACHE = {}


def gen(tok, model, convs, max_new, bs):
    """Generates in chunks of bs; on CUDA out-of-memory (long S5 histories), halves the chunk and retries."""
    out, i = [], 0
    while i < len(convs):
        chunk = convs[i:i + bs]
        try:
            out += _gen(tok, model, chunk, max_new)
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"OOM: batch -> {bs}", flush=True)
    return out


def _gen(tok, model, chunk, max_new):
    out = []
    if True:
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        out += [tok.decode(x[n:], skip_special_tokens=True).strip() for x in g]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/aya-expanse-8b"); ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    print("ready", a.model, a.adapter or "", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"ok": True, "model": a.model, "adapter": a.adapter})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                keepalive()
                t0 = time.time()
                with LOCK:
                    st = req.get("steer")
                    if st:
                        key = tuple(st)
                        if key not in VEC_CACHE:
                            VEC_CACHE[key] = {L: v.to(model.device) for L, v in steer_vectors(*key).items()}
                        with CAAModel(model, VEC_CACHE[key], strength=1.0):
                            r = gen(tok, model, req["convs"], req.get("max_new", 400), req.get("batch_size", 16))
                    else:
                        r = gen(tok, model, req["convs"], req.get("max_new", 400), req.get("batch_size", 16))
                keepalive()
                print(f"gen n={len(req['convs'])} steer={st} {time.time() - t0:.0f}s", flush=True)
                self._send(200, {"replies": r})
            except Exception as e:  # noqa: BLE001
                print("error", repr(e), flush=True)
                self._send(500, {"error": repr(e)})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
