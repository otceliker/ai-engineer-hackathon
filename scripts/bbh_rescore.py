#!/usr/bin/env python3
"""Re-score cached official generations with a ROBUST answer extractor (free — no re-gen).

The "answer is X" extractor misses lesson-induced phrasing ("the final result ... is True", "answer: X").
This tries several answer cues + a last-line fallback, then MC-letter / yes-no / free-form matching.
Reports base vs lessoned per task + overall, and how many old "regressions" were extraction artifacts.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "bbh", "cache")
TAG = "Qwen__Qwen2.5-7B-Instruct"
VARIANT = "failk20"

CUES = [r"answer is\s*:?\s*(.+)", r"answer:\s*(.+)",
        r"final (?:result|answer)[^.\n]*?\bis\s+(.+)", r"\bis\s*:?\s*(.+)$"]


def rextract(text):
    for pat in CUES:
        m = list(re.finditer(pat, text, re.IGNORECASE | re.MULTILINE))
        if m:
            return m[-1].group(1).strip()
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def _nf(s):
    return re.sub(r"\s+", " ", s.strip().lower().strip(".\"'`* )("))


def score(text, gold):
    post = rextract(text).rstrip(".").strip()
    g = gold.strip()
    if re.fullmatch(r"\([A-Za-z]\)", g):
        gl = g[1].lower()
        mm = re.search(r"\(([A-Za-z])\)", post) or re.match(r"([A-Za-z])\b", post)
        return bool(mm) and mm.group(1).lower() == gl
    if g.lower() in ("yes", "no"):
        mm = re.match(r"\s*(yes|no)\b", post, re.IGNORECASE)
        return bool(mm) and mm.group(1).lower() == g.lower()
    return _nf(post) == _nf(g) or _nf(g) == _nf(post.split(",")[-1])


def main():
    gen = json.load(open(os.path.join(CACHE, "official_gen.json")))
    rows = {r["key"]: r for r in json.load(open(os.path.join(ROOT, "data", "bbh", "bbh.json")))["rows"]}
    old = {}
    pj = os.path.join(ROOT, "results", "bbh", "official.jsonl")
    if os.path.exists(pj):
        for l in open(pj):
            if l.strip():
                r = json.loads(l)
                old[r["key"]] = r

    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0])   # base_ok, les_ok, n_base, n_les
    tot = [0, 0, 0, 0]
    artifact = 0
    for key, r in rows.items():
        bk, lk = f"base|{key}", f"les|{VARIANT}|{key}"
        if bk in gen:
            ok = score(gen[bk], r["target"]); agg[r["task"]][0] += ok; agg[r["task"]][2] += 1; tot[0] += ok; tot[2] += 1
        if lk in gen:
            ok = score(gen[lk], r["target"]); agg[r["task"]][1] += ok; agg[r["task"]][3] += 1; tot[1] += ok; tot[3] += 1
            o = old.get(key, {})
            if o.get("base_ok") and not o.get("les_ok") and ok:
                artifact += 1

    print(f"{'task':40} base      lessoned")
    for t in sorted(agg):
        a = agg[t]
        b = a[0] / a[2] if a[2] else 0
        l = a[1] / a[3] if a[3] else 0
        print(f"  {t:38} {b:.3f}({a[2]:2})  {l:.3f}({a[3]:2})  d={l-b:+.3f}")
    B = tot[0] / tot[2] if tot[2] else 0
    L = tot[1] / tot[3] if tot[3] else 0
    print("-" * 64)
    print(f"  {'OVERALL':38} {B:.3f}({tot[2]})  {L:.3f}({tot[3]})  delta={L-B:+.3f}")
    print(f"\nold 'regressions' that were extraction artifacts (now lessoned-correct): {artifact}")


if __name__ == "__main__":
    main()
