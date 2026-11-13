"""Blind closed-map coder with Claude Haiku 5.5 via the subscription (13 Nov 2026). Same DEF and lists as job.py."""
import json, sys, re, concurrent.futures as cf
sys.path.insert(0, "../heemeyer")
from ccbackend import cc_call
src = open("job.py").read()
DEF = re.search(r'DEF = """(.*?)"""', src, re.S).group(1)
opts = json.load(open("opts.json"))
def one(o):
    lst = "\n".join(f"- {x}" for x in o["options"]) or "(empty list)"
    try:
        txt, refused = cc_call("claude-haiku-5-5", DEF, "The list:\n" + lst)
    except Exception as e:
        return {"key": o["key"], "code": "error", "raw": str(e)[:200]}
    a = (txt or "").strip().strip("*").upper()
    code = "closed" if a.startswith("CLOSED") else "open" if a.startswith("OPEN") else "unparsed"
    return {"key": o["key"], "code": code, "raw": (txt or "")[:40]}
with cf.ThreadPoolExecutor(8) as ex, open("haiku_codes.jsonl", "w") as f:
    for i, r in enumerate(ex.map(one, opts)):
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("done", len(opts))
