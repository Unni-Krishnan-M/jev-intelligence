"""Evidence lineage: decision -> evidence -> run objects -> series -> data sources (Phase 2, P2.4).

``decision_lineage(result, decision_id)`` and ``warning_lineage(result, key)`` resolve, inside one
run result (``PipelineResult.to_dict()``), the graph behind a decision or a warning:

* the **root** (a decision; a warning links first to the ``early_warning_level`` decision it is
  downstream of, ``decision_id``);
* its **evidence** items, each with the object its ``ref`` points at;
* the **run objects** those refs name (signal, trend, anomaly, forecast, risk, decision, action),
  followed transitively through their own evidence refs and, for early-warning decisions, the
  ``state.components`` refs;
* the **series** each object is about (``series_id``, or a ref that is itself a series id);
* the **data sources** (``data.sources``): a series is built from the run's primary source (the first
  source row), an object with a ``source`` field names its source directly, and a decision or object
  with no resolvable object ref (a model-governance or run-level decision) is derived from the run's
  sources as a whole (``relation: "derived_from_run"``: the basis is the run, not one object).

``complete`` is true when every non-null ref resolved and at least one source is reached; ``unresolved``
lists the refs that did not. Pure and deterministic: the same result gives the same graph (nodes and
edges are sorted). The API adds the run row (mode, versions, events watermark) around it.
"""

from __future__ import annotations

from typing import Any

LINEAGE_VERSION = "lineage-1.0.0"
OBJECT_SECTIONS = {
    "signals": "signal",
    "trends": "trend",
    "anomalies": "anomaly",
    "risks": "risk",
    "decisions": "decision",
    "actions": "action",
}


def _index(result: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    idx: dict[str, tuple[str, dict[str, Any]]] = {}
    for sec, typ in OBJECT_SECTIONS.items():
        for o in result.get(sec) or []:
            if isinstance(o, dict) and o.get("id"):
                idx[str(o["id"])] = (typ, o)
    pred = result.get("predictions") or {}
    for f in (pred.get("forecasts") or []) + (pred.get("share_forecasts") or []):
        if isinstance(f, dict) and f.get("id"):
            idx[str(f["id"])] = ("forecast", f)
    return idx


def _title(typ: str, o: dict[str, Any]) -> str:
    if typ == "decision":
        return str(o.get("question") or o.get("key") or o.get("id"))
    if typ == "trend":
        return f"{o.get('series_id')} trend {o.get('direction')}"
    if typ == "forecast":
        return f"forecast {o.get('series_id')} ({o.get('model')})"
    if typ == "anomaly":
        return f"{o.get('kind')}: {o.get('series_id') or o.get('entity')}"
    return str(o.get("title") or o.get("key") or o.get("id"))


def _node(typ: str, oid: str, o: dict[str, Any]) -> dict[str, Any]:
    n: dict[str, Any] = {"id": f"{typ}:{oid}", "type": typ, "ref": oid, "title": _title(typ, o)[:300]}
    for k in ("kind", "key", "answer", "severity", "level", "direction", "confidence", "confidence_kind"):
        if o.get(k) is not None:
            n[k] = o[k]
    if o.get("series_id"):
        n["series_id"] = o["series_id"]
    return n


def _refs(typ: str, o: dict[str, Any]) -> list[str]:
    """Refs an object points at: evidence refs, plus early-warning component refs and a warning's
    source object."""
    out = [str(e["ref"]) for e in o.get("evidence") or [] if isinstance(e, dict) and e.get("ref")]
    if typ == "decision":
        for c in (o.get("state") or {}).get("components") or []:
            if isinstance(c, dict) and c.get("ref"):
                out.append(str(c["ref"]))
    return list(dict.fromkeys(out))


class _Graph:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.idx = _index(result)
        self.series = {str(s["id"]): s for s in result.get("series") or [] if isinstance(s, dict)}
        self.sources = [s for s in (result.get("data") or {}).get("sources") or [] if isinstance(s, dict)]
        self.source_names = {str(s.get("source")) for s in self.sources}
        self.primary = str(self.sources[0].get("source")) if self.sources else None
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: set[tuple[str, str, str]] = set()
        self.unresolved: list[dict[str, str]] = []

    def add(self, node: dict[str, Any]) -> str:
        self.nodes.setdefault(node["id"], node)
        return str(node["id"])

    def edge(self, a: str, b: str, rel: str) -> None:
        if a != b:
            self.edges.add((a, b, rel))

    def source(self, name: str) -> str:
        row = next((s for s in self.sources if str(s.get("source")) == name), {})
        node = {"id": f"source:{name}", "type": "source", "ref": name, "title": name}
        for k in ("kind", "rows", "first_event", "last_event", "fresh", "license", "url", "checksum"):
            if row.get(k) is not None:
                node[k] = row[k]
        return self.add(node)

    def series_node(self, sid: str) -> str:
        s = self.series[sid]
        nid = self.add(
            {
                "id": f"series:{sid}",
                "type": "series",
                "ref": sid,
                "title": f"{s.get('entity')} {s.get('metric')}",
                "unit": s.get("unit"),
                "frequency": s.get("frequency"),
                "n_points": len(s.get("points") or []),
            }
        )
        if self.primary is not None:
            self.edge(nid, self.source(self.primary), "built_from")
        return nid

    def run_sources(self, nid: str) -> None:
        for name in sorted(self.source_names):
            self.edge(nid, self.source(name), "derived_from_run")

    def visit(self, typ: str, oid: str, o: dict[str, Any], depth: int = 0) -> tuple[str, bool]:
        """Add the object and everything it depends on; returns (node id, reached a source)."""
        nid = f"{typ}:{oid}"
        if nid in self.nodes:
            return nid, True
        self.add(_node(typ, oid, o))
        reached = False
        sid = o.get("series_id")
        if isinstance(sid, str) and sid in self.series:
            self.edge(nid, self.series_node(sid), "about")
            reached = self.primary is not None
        src = o.get("source")
        if isinstance(src, str) and src in self.source_names:
            self.edge(nid, self.source(src), "observed_in")
            reached = True
        for ref in _refs(typ, o):
            if ref in self.idx and depth < 8:
                rt, ro = self.idx[ref]
                child, ok = self.visit(rt, ref, ro, depth + 1)
                self.edge(nid, child, "evidence")
                reached = reached or ok
            elif ref in self.series:
                self.edge(nid, self.series_node(ref), "evidence")
                reached = reached or self.primary is not None
            elif ref in self.source_names:
                self.edge(nid, self.source(ref), "evidence")
                reached = True
            else:
                self.unresolved.append({"from": nid, "ref": ref})
        if not reached and self.sources:
            self.run_sources(nid)
            reached = True
        return nid, reached

    def out(self, root: dict[str, Any], evidence: list[dict[str, Any]], reached: bool) -> dict[str, Any]:
        types = sorted({n["type"] for n in self.nodes.values()})
        return {
            "version": LINEAGE_VERSION,
            "root": root,
            "evidence": evidence,
            "nodes": sorted(self.nodes.values(), key=lambda n: (n["type"], n["id"])),
            "edges": [{"from": a, "to": b, "relation": r} for a, b, r in sorted(self.edges)],
            "series": sorted(n["ref"] for n in self.nodes.values() if n["type"] == "series"),
            "sources": sorted(n["ref"] for n in self.nodes.values() if n["type"] == "source"),
            "node_types": types,
            "unresolved": self.unresolved,
            "complete": reached and not self.unresolved,
        }

    def evidence_items(self, nid: str, o: dict[str, Any]) -> list[dict[str, Any]]:
        items = []
        for pos, e in enumerate(o.get("evidence") or []):
            if not isinstance(e, dict):
                continue
            ref = e.get("ref")
            resolved = None
            if ref:
                ref = str(ref)
                if ref in self.idx:
                    resolved = f"{self.idx[ref][0]}:{ref}"
                elif ref in self.series:
                    resolved = f"series:{ref}"
                elif ref in self.source_names:
                    resolved = f"source:{ref}"
            items.append(
                {
                    "owner": nid,
                    "position": pos,
                    "kind": e.get("kind"),
                    "label": e.get("label"),
                    "value": e.get("value"),
                    "detail": e.get("detail"),
                    "ref": ref,
                    "resolves_to": resolved,
                }
            )
        return items


def decision_lineage(result: dict[str, Any], decision_id: str) -> dict[str, Any] | None:
    """The lineage graph of one decision of ``result``; None when the run has no such decision."""
    g = _Graph(result)
    hit = g.idx.get(decision_id)
    if hit is None or hit[0] != "decision":
        return None
    d = hit[1]
    nid, reached = g.visit("decision", decision_id, d)
    root = {
        "type": "decision",
        "id": decision_id,
        "node": nid,
        "key": d.get("key"),
        "policy_version": d.get("policy_version"),
        "entity_type": d.get("entity_type"),
        "entity": d.get("entity"),
        "answer": d.get("answer"),
        "abstained": d.get("abstained"),
        "confidence": d.get("confidence"),
        "confidence_kind": d.get("confidence_kind"),
        "batch_id": d.get("batch_id"),
    }
    return g.out(root, g.evidence_items(nid, d), reached)


def warning_lineage(result: dict[str, Any], key: str) -> dict[str, Any] | None:
    """The lineage of the warning with dedup ``key`` in ``result``: warning -> its early-warning
    decision -> that decision's graph, plus the warning's own evidence and source object."""
    w = next((x for x in result.get("warnings") or [] if isinstance(x, dict) and x.get("key") == key), None)
    if w is None:
        return None
    g = _Graph(result)
    nid = g.add(
        {
            "id": f"warning:{key}",
            "type": "warning",
            "ref": key,
            "title": str(w.get("title") or key)[:300],
            "severity": w.get("severity"),
            "confidence": w.get("confidence"),
            "confidence_kind": w.get("confidence_kind"),
        }
    )
    reached = False
    dec_id = w.get("decision_id")
    if dec_id and dec_id in g.idx:
        child, ok = g.visit("decision", str(dec_id), g.idx[str(dec_id)][1])
        g.edge(nid, child, "decided_by")
        reached = ok
    elif dec_id:
        g.unresolved.append({"from": nid, "ref": str(dec_id)})
    src = (w.get("source") or {}).get("id")
    refs = [str(e["ref"]) for e in w.get("evidence") or [] if isinstance(e, dict) and e.get("ref")]
    for ref in dict.fromkeys(([str(src)] if src else []) + refs):
        if ref in g.idx:
            child, ok = g.visit(g.idx[ref][0], ref, g.idx[ref][1])
            g.edge(nid, child, "source" if ref == src else "evidence")
            reached = reached or ok
        elif ref in g.series:
            g.edge(nid, g.series_node(ref), "evidence")
            reached = reached or g.primary is not None
        else:
            g.unresolved.append({"from": nid, "ref": ref})
    if not reached and g.sources:
        g.run_sources(nid)
        reached = True
    root = {
        "type": "warning",
        "id": key,
        "node": nid,
        "key": key,
        "decision_id": dec_id,
        "early_warning_level": w.get("early_warning_level"),
        "severity": w.get("severity"),
        "source": w.get("source"),
    }
    out = g.out(root, g.evidence_items(nid, w), reached)
    out["complete"] = out["complete"] and bool(dec_id) and dec_id in g.idx
    return out


__all__ = ["LINEAGE_VERSION", "decision_lineage", "warning_lineage"]
