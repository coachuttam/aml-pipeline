"""Checks on a goAML-style draft: our own schema, and the list of missing fields.

`validation_errors` checks the draft against docs/goaml_style_draft.xsd (written for
this project, not a certified UAE FIU schema). `missing_fields` lists every
"MISSING: <what>" placeholder, grouped, so the UI can show what the analyst or
compliance officer must still fill in before filing.

Example:
    validation_errors(xml)   # [] or ["line 12: Element 'reason': ..."]
    missing_fields(xml)      # [{"field": "SWIFT/BIC", "where": [...], "count": 6}]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src import config


def validation_errors(xml: bytes, schema_path: Path = config.GOAML_SCHEMA_FILE):
    """Schema errors as readable lines (empty list: the draft fits the schema)."""
    from lxml import etree

    schema = etree.XMLSchema(etree.parse(str(schema_path)))
    document = etree.fromstring(xml)
    if schema.validate(document):
        return []
    return ["line %d: %s" % (e.line, e.message) for e in schema.error_log]


def element_path(element, root) -> str:
    """ "transaction/t_from/from_account/swift" (no positions, so they group)."""
    names = [element.tag]
    for ancestor in element.iterancestors():
        if ancestor is root:
            break
        names.append(ancestor.tag)
    return "/".join(reversed(names))


def missing_fields(xml: bytes) -> List[Dict]:
    """Each placeholder kind once: what is missing, where, and how many times."""
    from lxml import etree

    root = etree.fromstring(xml)
    found: Dict[str, Dict] = {}
    for element in root.iter():
        text = element.text or ""
        if not isinstance(element.tag, str) or not text.startswith(
            config.GOAML_MISSING_PREFIX
        ):
            continue
        what = text[len(config.GOAML_MISSING_PREFIX) :]
        entry = found.setdefault(what, {"field": what, "where": [], "count": 0})
        entry["count"] += 1
        path = element_path(element, root)
        if path not in entry["where"]:
            entry["where"].append(path)
    return sorted(found.values(), key=lambda e: e["field"])
