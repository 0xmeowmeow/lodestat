"""OpenAI-compatible chat backend for persona models (9 Nov 2026).

Spec: 'oai:<base_url>|<model>', e.g.
    oai:https://router.huggingface.co/v1|<org>/<model>     (Hugging Face router; uses the local HF token)
    oai:https://<endpoint>.endpoints.huggingface.cloud/v1|tgi   (a dedicated HF Inference Endpoint)
    oai:http://127.0.0.1:8902/v1|<model>                   (vLLM on the rented GPU, via the SSH tunnel)
The key comes from $OAI_API_KEY, else the local Hugging Face token file. It is never printed or logged.
"""
import json, os, pathlib, time, urllib.error, urllib.request


def _key():
    if os.environ.get("OAI_API_KEY"):
        return os.environ["OAI_API_KEY"]
    p = pathlib.Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else ""


def oai_call(base, model, system, user, seed=0, temperature=0.9, max_tokens=1200, tries=5):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:   # cold start / rate limit
                time.sleep(15 * (i + 1)); continue
            raise RuntimeError(f"oai {e.code}: {e.read()[:200]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
            time.sleep(15 * (i + 1))
    return ""
