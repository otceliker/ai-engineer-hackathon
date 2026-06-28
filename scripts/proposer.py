#!/usr/bin/env python3
"""Pluggable proposer: reads agent failures, proposes edits to the lessons playbook.

Two backends, one interface — `propose(failures, lessonbook) -> [edit, ...]`:
  LocalProposer  — the same 1.5B agent proposes its own lessons (the pure-RSI
                   baseline arm; reuses the agent's vLLM engine, no second load).
  ClaudeProposer — a Claude model (default claude-opus-4-8) proposes lessons (the
                   upgrade arm). Reads ANTHROPIC_API_KEY from the environment.

Both share the failure-formatting and the tolerant edit parser, so swapping the
backend is the only variable — the proposer swap is itself a measurable result.

Edit schema (validated downstream by lessons.LessonBook.apply):
    {"op": "add", "text": "..."}
    {"op": "replace", "id": <int>, "text": "..."}
    {"op": "remove", "id": <int>}
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Default proposer model for the Claude arm. The claude-api reference is explicit:
# use claude-opus-4-8 unless the user names another model.
CLAUDE_MODEL = os.environ.get("PROPOSER_CLAUDE_MODEL", "claude-opus-4-8")

PROPOSER_SYSTEM = (
    "You improve a small function-calling agent by maintaining a SHORT playbook of "
    "general lessons that are appended to its system prompt. You are shown the current "
    "playbook and a batch of the agent's FAILED tasks (the user request, the available "
    "tools, what the agent emitted, and what the correct behavior was).\n\n"
    "Propose minimal edits so the agent avoids these mistakes in the future.\n"
    "Rules for good lessons:\n"
    "- General and reusable — describe a behavior pattern, never a single task's answer.\n"
    "- Imperative and short (one sentence).\n"
    "- Target the actual failure mode you see in the batch.\n"
    "- The agent is small; too many or too vague lessons HURT it. Prefer 0-2 edits per round.\n"
    "- The playbook is hard-capped. To add past the cap you MUST remove or replace an "
    "existing lesson (reference it by its [id]).\n\n"
    "Output ONLY a JSON object of the form "
    '{"edits": [{"op": "add", "text": "..."}, {"op": "replace", "id": 2, "text": "..."}, '
    '{"op": "remove", "id": 5}]}. '
    "If the current lessons already cover these failures, output {\"edits\": []}."
)

# JSON schema for the Claude arm (structured outputs). All props required + sentinels,
# per the structured-outputs constraint that every property appear in `required`.
_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "replace", "remove"]},
                    "id": {"type": "integer", "description": "lesson id for replace/remove; -1 for add"},
                    "text": {"type": "string", "description": "new text for add/replace; empty for remove"},
                },
                "required": ["op", "id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------- format
def _summarize_ground_truth(record):
    kind = record["kind"]
    if kind == "irrelevance":
        return "NO function call — none of the available tools can satisfy this request."
    if kind == "relevance":
        return "At least ONE valid function call (a relevant tool exists)."
    parts = []
    for call in record.get("ground_truth") or []:
        for name, args in call.items():
            shown = {k: (v[0] if isinstance(v, list) and v else v) for k, v in args.items()}
            parts.append(f"{name}({', '.join(f'{k}={v!r}' for k, v in shown.items())})")
    return " AND ".join(parts) if parts else "(none)"


def format_failure(record, raw_output, parsed_calls):
    """Compact, model-readable description of one failed task."""
    user_msgs = [m["content"] for m in record["question"][0] if m.get("role") == "user"]
    query = " ".join(user_msgs)[:400]
    tools = ", ".join(f.get("name", "?") for f in record["function"])[:300]
    got = json.dumps([{c["name"]: c["arguments"]} for c in parsed_calls]) if parsed_calls else "(no call)"
    return (
        f"- category: {record['category']}\n"
        f"  request: {query}\n"
        f"  tools: {tools}\n"
        f"  agent emitted: {got[:300]}\n"
        f"  correct: {_summarize_ground_truth(record)}"
    )


def build_user_message(lessonbook, failures, max_show=10):
    shown = failures[:max_show]
    blocks = "\n".join(format_failure(*f) for f in shown)
    more = f"\n(and {len(failures) - max_show} more failures this round)" if len(failures) > max_show else ""
    cap = (f"Playbook caps: max {lessonbook.max_bullets} lessons, "
           f"~{lessonbook.max_chars} chars total, {lessonbook.max_lesson_chars} chars/lesson. "
           f"Currently {len(lessonbook.lessons)} lessons, {lessonbook.total_chars()} chars.")
    return (
        f"Current playbook:\n{lessonbook.render_for_proposer()}\n\n"
        f"{cap}\n\n"
        f"Failed tasks this round:\n{blocks}{more}\n\n"
        "Propose edits as JSON."
    )


# ----------------------------------------------------------------------------- parse
def parse_edits(text):
    """Tolerantly extract a list of edit dicts from model output (either backend)."""
    obj = _first_json(text)
    edits = []
    if isinstance(obj, dict) and isinstance(obj.get("edits"), list):
        raw = obj["edits"]
    elif isinstance(obj, list):
        raw = obj
    else:
        return []
    for e in raw:
        if not isinstance(e, dict) or "op" not in e:
            continue
        op = e["op"]
        out = {"op": op}
        # carry id only when meaningful (sentinel -1 / missing -> drop)
        if op in ("replace", "remove") and isinstance(e.get("id"), int) and e["id"] >= 0:
            out["id"] = e["id"]
        if op in ("add", "replace"):
            out["text"] = e.get("text", "")
        edits.append(out)
    return edits


def _first_json(text):
    """Find the first JSON object or array in text and parse it."""
    # Try whichever bracket type appears first, so a bare array also parses.
    candidates = sorted(
        ((text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if o in text),
        key=lambda t: t[0],
    )
    for start, opener, closer in candidates:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except Exception:
                            break
    return None


# ----------------------------------------------------------------------------- backends
class LocalProposer:
    """The agent's own 1.5B model proposes lessons (pure-RSI baseline)."""

    def __init__(self, agent, temperature=0.7, max_tokens=512):
        self.agent = agent
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose(self, failures, lessonbook):
        if not failures:
            return []
        messages = [
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content": build_user_message(lessonbook, failures)},
        ]
        text = self.agent.complete(messages, max_tokens=self.max_tokens,
                                   temperature=self.temperature)
        return parse_edits(text)


class ClaudeProposer:
    """A Claude model proposes lessons (upgrade arm). Needs ANTHROPIC_API_KEY."""

    def __init__(self, model=CLAUDE_MODEL, max_tokens=2048):
        try:
            import anthropic
        except ImportError:
            sys.exit("Missing dependency: anthropic. Run: uv pip install anthropic")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def propose(self, failures, lessonbook):
        if not failures:
            return []
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=PROPOSER_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _EDIT_SCHEMA}},
            messages=[{"role": "user", "content": build_user_message(lessonbook, failures)}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return parse_edits(text)


class OpenAICompatProposer:
    """Proposer over any OpenAI-compatible chat endpoint (e.g. DigitalOcean GenAI
    serving Claude Opus 4.8). Config via env:
        PROPOSER_BASE_URL  (default https://inference.do-ai.run/v1)
        PROPOSER_MODEL     (default anthropic-claude-opus-4.8)
        DO_TOKEN | PROPOSER_API_KEY  (bearer key)
    Exponential backoff on 429/5xx; raises after N consecutive failures rather
    than silently degrading the run to no-ops (per repo error-handling rules)."""

    def __init__(self, base_url=None, model=None, api_key=None, max_tokens=2048,
                 temperature=0.3, max_retries=5):
        try:
            import requests  # noqa: F401
        except ImportError:
            sys.exit("Missing dependency: requests. Run: uv pip install requests")
        self.base_url = (base_url or os.environ.get("PROPOSER_BASE_URL")
                         or "https://inference.do-ai.run/v1").rstrip("/")
        self.model = model or os.environ.get("PROPOSER_MODEL") or "anthropic-claude-opus-4.8"
        self.api_key = (api_key or os.environ.get("PROPOSER_API_KEY")
                        or os.environ.get("DO_TOKEN"))
        if not self.api_key:
            sys.exit("OpenAICompatProposer needs an API key (DO_TOKEN or PROPOSER_API_KEY).")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def propose(self, failures, lessonbook):
        if not failures:
            return []
        import time
        import requests
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": PROPOSER_SYSTEM},
                {"role": "user", "content": build_user_message(lessonbook, failures)},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        delay = 2
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=120)
            except requests.RequestException as e:
                last = f"network error: {e}"
            else:
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    return parse_edits(text)
                # 4xx other than 429 are not retryable — fail loud
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Proposer API {resp.status_code}: {resp.text[:300]}")
                last = f"HTTP {resp.status_code}"
            if attempt < self.max_retries:
                print(f"  [proposer] {last}; backoff {delay}s ({attempt}/{self.max_retries})", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError(f"Proposer API failed after {self.max_retries} attempts ({last}).")


def make_proposer(backend, agent=None, **kwargs):
    if backend == "local":
        if agent is None:
            raise ValueError("LocalProposer needs the shared BfclAgent")
        return LocalProposer(agent, **kwargs)
    if backend == "claude":
        return ClaudeProposer(**kwargs)
    if backend == "do":  # OpenAI-compatible (DigitalOcean GenAI, etc.)
        return OpenAICompatProposer(**kwargs)
    raise ValueError(f"Unknown proposer backend: {backend!r}")
