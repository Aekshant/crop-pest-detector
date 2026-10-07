#!/usr/bin/env python3
"""Laya tool schema v1: load, check, and build the UNDERSTAND output schema.

  python ml/bundle/tool_schema.py            # check the schema file
  python ml/bundle/tool_schema.py --print    # print the UNDERSTAND JSON Schema

UNDERSTAND (design doc, Laya state machine) outputs exactly one of:
  {"intent": str, "tool": <tool>, "args": {...}, "reply": str}     call a tool
  {"clarify": str}                                                 ask one question
  {"intent": <respond kind>, "reply": str}                         answer directly
The JSON Schema built here is used to validate outputs and, in L4, handed to
llama.cpp for constrained (grammar) decoding.
"""

import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
SCHEMA_FILE = BUNDLE / "tools_schema_v1.json"
LABELS_FILE = BUNDLE.parent / "labels_v1.json"

ARG_TYPES = {"string", "integer", "label", "datetime"}
# ISO 8601 date-time with seconds and an explicit offset (the app works in +05:30).
DATETIME_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)$"


def load(path=SCHEMA_FILE):
    return json.loads(Path(path).read_text())


def check(schema):
    """Return a list of problems with the schema file (empty when fine)."""
    errors = []
    if not isinstance(schema.get("schema_version"), int):
        errors.append("schema_version must be an integer")
    v1 = [l["id"] for l in json.loads(LABELS_FILE.read_text())["labels"]]
    if schema.get("labels") != [l for l in v1 if l != "unknown"]:
        errors.append("labels must equal ml/labels_v1.json ids minus 'unknown', in order")
    if not schema.get("respond_kinds"):
        errors.append("respond_kinds missing")
    for name, tool in schema.get("tools", {}).items():
        for key in ("description", "needsNetwork", "sideEffect", "args"):
            if key not in tool:
                errors.append(f"{name}: missing {key}")
        for arg, spec in tool.get("args", {}).items():
            if spec.get("type") not in ARG_TYPES:
                errors.append(f"{name}.{arg}: unknown type {spec.get('type')}")
            if not isinstance(spec.get("required"), bool):
                errors.append(f"{name}.{arg}: required must be true/false")
    return errors


def _arg_schema(spec, labels):
    t = spec["type"]
    if t == "label":
        return {"type": "string", "enum": labels}
    if t == "datetime":
        return {"type": "string", "pattern": DATETIME_PATTERN}
    if t == "integer":
        s = {"type": "integer"}
        for k in ("minimum", "maximum"):
            if k in spec:
                s[k] = spec[k]
        return s
    return {"type": "string", "minLength": 1}


def understand_schema(schema, tools=None):
    """JSON Schema for one UNDERSTAND output. `tools` limits the allowed tools
    (e.g. without network tools when OFFLINE)."""
    labels = schema["labels"]
    variants = []
    for name, tool in schema["tools"].items():
        if tools is not None and name not in tools:
            continue
        args = tool["args"]
        variants.append({
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "tool": {"const": name},
                "args": {
                    "type": "object",
                    "properties": {a: _arg_schema(s, labels) for a, s in args.items()},
                    "required": [a for a, s in args.items() if s["required"]],
                    "additionalProperties": False,
                },
                "reply": {"type": "string"},
            },
            "required": ["intent", "tool", "args", "reply"],
            "additionalProperties": False,
        })
    variants.append({
        "type": "object",
        "properties": {"clarify": {"type": "string", "minLength": 1}},
        "required": ["clarify"],
        "additionalProperties": False,
    })
    variants.append({
        "type": "object",
        "properties": {"intent": {"enum": schema["respond_kinds"]}, "reply": {"type": "string", "minLength": 1}},
        "required": ["intent", "reply"],
        "additionalProperties": False,
    })
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"Laya UNDERSTAND output (tool schema v{schema['schema_version']})",
            "oneOf": variants}


def main():
    schema = load()
    errors = check(schema)
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    if "--print" in sys.argv:
        print(json.dumps(understand_schema(schema), indent=2, ensure_ascii=False))
    elif not errors:
        print(f"tool schema v{schema['schema_version']}: {len(schema['tools'])} tools, {len(schema['labels'])} labels, ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
