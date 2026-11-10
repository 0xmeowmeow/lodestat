"""Grok as a persona or rater via the Grok CLI's single-turn mode (`grok -p`), billed to the user's
Grok subscription, not the xAI API. Spec in the runners: 'xai:<model>' (e.g. xai:grok-4.7).

Login once:  grok login     (models: grok models)
Isolation mirrors ccbackend: empty working dir, system prompt replaced, no tools, no web, no subagents,
any XAI_API_KEY removed from the child env so it can't fall back to API billing.
Single-turn only.
"""
import json, os, subprocess, tempfile

_WD = tempfile.mkdtemp(prefix="grokwd-")
_ENV = {k: v for k, v in os.environ.items() if k not in ("XAI_API_KEY", "GROK_API_KEY")}
# Clean HOME holding only Grok's login: otherwise the CLI also loads ~/.claude/CLAUDE.md and the user's skills.
_HOME = tempfile.mkdtemp(prefix="grokhome-")
os.symlink(os.path.expanduser("~/.grok"), os.path.join(_HOME, ".grok"))
_ENV["HOME"] = _HOME
USAGE = {"calls": 0}


def xai_call(model, system, user, temperature=None, timeout=600):
    """Returns (text, refused)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=_WD, delete=False) as f:
        f.write(user)
        pf = f.name
    try:
        p = subprocess.run(
            ["grok", "--model", model, "--reasoning-effort", "low", "--prompt-file", pf, "--system-prompt-override", system,
             "--tools", "", "--disable-web-search", "--no-subagents", "--no-plan", "--max-turns", "1",
             "--verbatim", "--output-format", "json"],
            capture_output=True, text=True, cwd=_WD, env=_ENV, timeout=timeout, stdin=subprocess.DEVNULL)
    finally:
        os.unlink(pf)
    USAGE["calls"] += 1
    out = p.stdout.strip()
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        if p.returncode != 0 or not out:
            raise RuntimeError(f"grok failed (exit {p.returncode}): {(p.stderr or out)[-500:]}")
        return out, False  # plain text fallback
    refused_flag = d.get("stopReason") == "refusal"
    text = d.get("result") or d.get("text") or d.get("content") or ""
    if isinstance(text, list):
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    refused = refused_flag or d.get("stop_reason") == "refusal" or d.get("finish_reason") == "content_filter"
    if d.get("is_error") and not refused:
        raise RuntimeError(f"grok error: {str(d)[:500]}")
    return ("" if refused else text), refused
