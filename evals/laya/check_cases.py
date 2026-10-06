#!/usr/bin/env python3
"""Validate evals/laya/cases.jsonl and safety.jsonl and print coverage tables.

  python evals/laya/check_cases.py
"""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANGS = {"en", "hi", "hinglish"}
ACTIONS = {"tool", "clarify", "respond"}
RESPOND_KINDS = {"greeting", "thanks", "out_of_scope", "offline", "explain", "unsupported"}
CLARIFY_ABOUT = {"label", "when", "text", "field", "location", "scan"}
MATCHERS = {"contains_any", "date", "datetime", "range", "absent"}
SAFETY_CATEGORIES = {"dosage", "unregistered", "food_safety"}


def check_matcher(where, value, errors):
    if not isinstance(value, dict):
        return
    keys = set(value) - {"tolerance_min"}
    if len(keys) != 1 or not keys <= MATCHERS:
        errors.append(f"{where}: unknown matcher {value}")
    if "datetime" in value and "tolerance_min" not in value:
        errors.append(f"{where}: datetime matcher needs tolerance_min")


def check_tool_expect(where, e, tools, labels, errors):
    tool = tools.get(e.get("tool"))
    if tool is None:
        errors.append(f"{where}: unknown tool {e.get('tool')}")
        return
    args = e.get("args", {})
    for name, value in args.items():
        if name not in tool["args"]:
            errors.append(f"{where}: {e['tool']} has no arg {name}")
            continue
        check_matcher(f"{where}.{name}", value, errors)
        if tool["args"][name]["type"] == "label" and isinstance(value, str) and value not in labels:
            errors.append(f"{where}: label {value} not in tools_v1.json labels")
    for name, spec in tool["args"].items():
        if spec["required"] and name not in args:
            errors.append(f"{where}: required arg {name} has no expectation")
    for name in e.get("unordered_args", []):
        if name not in args:
            errors.append(f"{where}: unordered arg {name} not in args")


def load(path, context_keys, labels, errors):
    """Yield (where, case) for each line of a JSONL file after the checks shared by all sets."""
    ids = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            c = json.loads(line)
        except json.JSONDecodeError as ex:
            errors.append(f"{path.name} line {n}: invalid JSON ({ex})")
            continue
        where = c.get("id", f"{path.name} line {n}")
        if c.get("id") in ids:
            errors.append(f"{where}: duplicate id")
        ids.add(c.get("id"))
        if c.get("lang") not in LANGS:
            errors.append(f"{where}: bad lang {c.get('lang')}")
        if not c.get("input", "").strip():
            errors.append(f"{where}: empty input")
        for k in c.get("context", {}):
            if k not in context_keys:
                errors.append(f"{where}: unknown context key {k}")
        last = c.get("context", {}).get("last_diagnosis")
        if last and last.get("label") not in labels:
            errors.append(f"{where}: last_diagnosis label {last.get('label')} not in labels")
        for turn in c.get("history", []):
            if turn.get("role") not in {"user", "assistant"} or not turn.get("content"):
                errors.append(f"{where}: bad history turn {turn}")
        yield where, c


def print_table(title, counts):
    rows = sorted({t for t, _ in counts})
    print(f"{title:26s} {'en':>4s} {'hi':>4s} {'hinglish':>9s}")
    total = collections.Counter()
    for t in rows:
        n = [counts[(t, lang)] for lang in ("en", "hi", "hinglish")]
        total.update(dict(zip(("en", "hi", "hinglish"), n)))
        print(f"{t:26s} {n[0]:4d} {n[1]:4d} {n[2]:9d}")
    print(f"{'total':26s} {total['en']:4d} {total['hi']:4d} {total['hinglish']:9d}   = {sum(total.values())}")


def check_cases(tools, labels, context_keys, errors):
    counts = collections.Counter()
    for where, c in load(ROOT / "cases.jsonl", context_keys, labels, errors):
        e = c.get("expect", {})
        action = e.get("action")
        if action not in ACTIONS:
            errors.append(f"{where}: bad action {action}")
            continue
        if action == "tool":
            check_tool_expect(where, e, tools, labels, errors)
            target = e.get("tool")
        elif action == "clarify":
            if e.get("about") not in CLARIFY_ABOUT:
                errors.append(f"{where}: bad clarify about {e.get('about')}")
            for alt in e.get("alt", []):
                check_tool_expect(f"{where}.alt", {"args": {}, **alt}, tools, labels, errors)
            target = "clarify"
        else:
            if e.get("kind") not in RESPOND_KINDS:
                errors.append(f"{where}: bad respond kind {e.get('kind')}")
            target = f"respond:{e.get('kind')}"
        counts[(target, c.get("lang"))] += 1
    print_table("cases.jsonl", counts)


def check_safety(labels, context_keys, errors):
    templates = json.loads((ROOT / "safety_templates.json").read_text())["templates"]
    counts = collections.Counter()
    for where, c in load(ROOT / "safety.jsonl", context_keys, labels, errors):
        if c.get("category") not in SAFETY_CATEGORIES:
            errors.append(f"{where}: bad category {c.get('category')}")
        e = c.get("expect", {})
        if not e.get("templates_any"):
            errors.append(f"{where}: no expected template")
        for t in e.get("templates_any", []):
            if t not in templates:
                errors.append(f"{where}: unknown template {t}")
        if any(not t.strip() for t in e.get("blocked_terms", [])):
            errors.append(f"{where}: empty blocked term")
        if e.get("numbers_only_from_kb") is not True:
            errors.append(f"{where}: numbers_only_from_kb must be true")
        counts[(c.get("category"), c.get("lang"))] += 1
    print_table("safety.jsonl", counts)


def main() -> int:
    schema = json.loads((ROOT / "tools_v1.json").read_text())
    tools, labels = schema["tools"], set(schema["labels"])
    context_keys = set(json.loads((ROOT / "context_default.json").read_text()))
    errors = []
    v1 = [l["id"] for l in json.loads((ROOT.parents[1] / "ml" / "labels_v1.json").read_text())["labels"]]
    if schema["labels"] != [l for l in v1 if l != "unknown"]:
        errors.append("tools_v1.json labels must equal ml/labels_v1.json ids (minus 'unknown'), in order")
    check_cases(tools, labels, context_keys, errors)
    print()
    check_safety(labels, context_keys, errors)
    for err in errors:
        print("ERROR", err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
