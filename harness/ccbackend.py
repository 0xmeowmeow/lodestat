"""Claude calls routed through Claude Code headless mode (`claude -p`) so they bill the user's
Claude subscription (and its extra-usage credit) instead of the API Console balance.

Spec prefix in the runners: 'cc:<model>' (e.g. cc:claude-sonnet-5-5) instead of 'anthropic:<model>'.

Isolation (checked 8 Nov 2026): --system-prompt-file replaces Claude Code's own system prompt,
CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 stops the user's CLAUDE.md loading, --tools "" removes tools,
--setting-sources "" and --strict-mcp-config skip settings/hooks/MCP. ANTHROPIC_API_KEY is removed
from the child env so auth falls through to the subscription login (with the key set, it would bill
the API). Claude Code still attaches the account email as context; nothing else leaked in the test.
Single-turn only: one system prompt + one user message.
"""
import time, json
import os
import subprocess
import tempfile

_WD = tempfile.mkdtemp(prefix="ccwd-")  # empty cwd: no project CLAUDE.md to discover
USAGE = {"calls": 0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "list_usd": 0.0}
_ENV = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
_ENV["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"


NET_ERR = ("Can't reach the API", "EAI_AGAIN", "ECONNRESET", "ETIMEDOUT", "overloaded", "Overloaded", "529", "rate limit",
           "Rate limit", "ENOTFOUND", "socket hang up", "fetch failed")


def cc_call(model, system, user, effort="low", timeout=600):
    """Retries network outages and overload for up to ~20 min (the 9 Nov wifi drop killed a whole night's runs)."""
    for i in range(24):
        try:
            return _cc_call(model, system, user, effort, timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            if i == 23 or not (isinstance(e, subprocess.TimeoutExpired) or any(k in str(e) for k in NET_ERR)):
                raise
            time.sleep(min(60, 10 * (i + 1)))


def _cc_call(model, system, user, effort="low", timeout=600):
    """Returns (text, refused: bool). text is '' on refusal or error."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=_WD, delete=False) as f:
        f.write(system)
        sp = f.name
    try:
        for _try in range(30):  # Claude Code auto-updates swap the binary; it can vanish for a few seconds (8 Nov S6 crash)
            try:
                p = _run(sp, model, effort, user, timeout); break
            except (FileNotFoundError, PermissionError, OSError):  # incl. Errno 8 Exec format error: half-written binary (9 Nov a1-ask)
                if _try == 29: raise
                time.sleep(10)
    finally:
        os.unlink(sp)
    return _parse(p)


def _run(sp, model, effort, user, timeout):
    return subprocess.run(
            ["claude", "-p", "--model", model, "--effort", effort, "--system-prompt-file", sp,
             "--tools", "", "--setting-sources", "", "--strict-mcp-config",
             "--no-session-persistence", "--output-format", "json"],
            input=user, capture_output=True, text=True, cwd=_WD, env=_ENV, timeout=timeout)


def _parse(p):
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude -p failed (exit {p.returncode}): {p.stderr[-500:] or p.stdout[-500:]}")
    u = d.get("usage") or {}
    USAGE["calls"] += 1
    USAGE["input"] += u.get("input_tokens", 0); USAGE["output"] += u.get("output_tokens", 0)
    USAGE["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
    USAGE["cache_write"] += u.get("cache_creation_input_tokens", 0) or 0
    USAGE["list_usd"] += d.get("total_cost_usd") or 0.0
    refused = d.get("stop_reason") == "refusal"
    if d.get("is_error") and not refused:
        raise RuntimeError(f"claude -p error: {str(d.get('result'))[:500]}")
    return ("" if refused else (d.get("result") or "")), refused
