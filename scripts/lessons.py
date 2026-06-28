#!/usr/bin/env python3
"""Bounded lessons playbook for the streaming prompt-patching loop.

A `LessonBook` is a short, curated list of imperative guidance bullets that get
appended to the agent's base system prompt. The proposer edits it via structured
ops (add / replace / remove). The book is *hard-capped* — both in bullet count and
total characters — so a small model never drowns in a bloated prompt (the failure
mode that sank STaR rationalization: see docs/WIKI.md). To add past the cap the
proposer MUST remove something first; over-cap adds are rejected, not silently
truncated.

Each lesson has a stable integer id so replace/remove can target it across rounds.

Edit schema (what the proposer emits, validated here):
    {"op": "add", "text": "..."}
    {"op": "replace", "id": 3, "text": "..."}
    {"op": "remove", "id": 5}
"""
import json


class LessonBook:
    def __init__(self, max_bullets=10, max_chars=800, max_lesson_chars=160):
        self.max_bullets = max_bullets
        self.max_chars = max_chars
        self.max_lesson_chars = max_lesson_chars
        self.lessons = []          # list of {"id": int, "text": str}
        self._next_id = 0

    # ---- state ---------------------------------------------------------------
    def texts(self):
        return [l["text"] for l in self.lessons]

    def ids(self):
        return [l["id"] for l in self.lessons]

    def total_chars(self):
        return sum(len(l["text"]) for l in self.lessons)

    def is_empty(self):
        return not self.lessons

    def copy(self):
        nb = LessonBook(self.max_bullets, self.max_chars, self.max_lesson_chars)
        nb.lessons = [dict(l) for l in self.lessons]
        nb._next_id = self._next_id
        return nb

    # ---- rendering for the proposer ------------------------------------------
    def render_for_proposer(self):
        """Numbered by stable id, so the proposer can reference ids in edits."""
        if not self.lessons:
            return "(empty)"
        return "\n".join(f"[{l['id']}] {l['text']}" for l in self.lessons)

    # ---- edits ---------------------------------------------------------------
    def apply(self, edits):
        """Apply a list of edits. Returns (applied, rejected) where each rejected
        entry is (edit, reason). Caps are enforced as hard constraints."""
        applied, rejected = [], []
        for e in edits:
            ok, reason = self._apply_one(e)
            (applied if ok else rejected).append(e if ok else (e, reason))
        return applied, rejected

    def _find(self, lid):
        for i, l in enumerate(self.lessons):
            if l["id"] == lid:
                return i
        return None

    def _clean(self, text):
        return " ".join(str(text).split()).strip()

    def _apply_one(self, e):
        if not isinstance(e, dict) or "op" not in e:
            return False, "malformed edit"
        op = e["op"]

        if op == "remove":
            i = self._find(e.get("id"))
            if i is None:
                return False, f"remove: no lesson id {e.get('id')}"
            self.lessons.pop(i)
            return True, ""

        if op == "replace":
            i = self._find(e.get("id"))
            if i is None:
                return False, f"replace: no lesson id {e.get('id')}"
            text = self._clean(e.get("text", ""))
            if not text:
                return False, "replace: empty text"
            if len(text) > self.max_lesson_chars:
                return False, f"replace: lesson > {self.max_lesson_chars} chars"
            old = self.lessons[i]["text"]
            self.lessons[i]["text"] = text
            if self.total_chars() > self.max_chars:
                self.lessons[i]["text"] = old  # roll back
                return False, "replace: would exceed total char cap"
            return True, ""

        if op == "add":
            text = self._clean(e.get("text", ""))
            if not text:
                return False, "add: empty text"
            if len(text) > self.max_lesson_chars:
                return False, f"add: lesson > {self.max_lesson_chars} chars"
            if any(l["text"].lower() == text.lower() for l in self.lessons):
                return False, "add: duplicate"
            if len(self.lessons) >= self.max_bullets:
                return False, f"add: at bullet cap ({self.max_bullets}); remove one first"
            if self.total_chars() + len(text) > self.max_chars:
                return False, "add: would exceed total char cap"
            self.lessons.append({"id": self._next_id, "text": text})
            self._next_id += 1
            return True, ""

        return False, f"unknown op '{op}'"

    # ---- serialization -------------------------------------------------------
    def to_dict(self):
        return {
            "max_bullets": self.max_bullets,
            "max_chars": self.max_chars,
            "max_lesson_chars": self.max_lesson_chars,
            "next_id": self._next_id,
            "lessons": self.lessons,
        }

    @classmethod
    def from_dict(cls, d):
        nb = cls(d["max_bullets"], d["max_chars"], d.get("max_lesson_chars", 160))
        nb.lessons = list(d["lessons"])
        nb._next_id = d["next_id"]
        return nb

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    # Self-test: exercise caps, ids, edit ops.
    b = LessonBook(max_bullets=3, max_chars=120, max_lesson_chars=60)
    ap, rj = b.apply([
        {"op": "add", "text": "Do not call a tool unless its required params are satisfiable."},
        {"op": "add", "text": "Emit one tool call per independent action."},
        {"op": "add", "text": "duplicate"}, {"op": "add", "text": "Duplicate"},
    ])
    print("after adds:", b.texts())
    print("rejected:", [(r[0].get('text', r[0]), r[1]) for r in rj])
    # bullet cap (3): a 3rd unique add ok, 4th rejected
    ap2, rj2 = b.apply([{"op": "add", "text": "Third short lesson."},
                        {"op": "add", "text": "Fourth should be rejected at cap."}])
    print("ids:", b.ids(), "| over-cap rejected:", [r[1] for r in rj2])
    # replace + remove by id
    b.apply([{"op": "replace", "id": 0, "text": "No call when params unsatisfiable."}])
    b.apply([{"op": "remove", "id": 1}])
    print("final:", b.render_for_proposer())
    print("total_chars:", b.total_chars(), "<=", b.max_chars)
