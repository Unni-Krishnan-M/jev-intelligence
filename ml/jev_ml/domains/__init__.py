"""Domain adapters and their registry.

``available()`` lists every adapter this installation knows: ``movie`` and one
``generic:<name>`` per ``configs/domains/<name>.yaml``, each with ``available`` (can it load on this
machine: the movie adapter needs ``data/processed``, a generic adapter its dataset file) and a
``reason`` when not. ``get_adapter(key)`` builds one; call it with ``jev_ml.core.run_domain``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _root(root: Path | None) -> Path:
    from jev_ml.paths import ROOT

    return root or ROOT


def available(root: Path | None = None) -> list[dict[str, Any]]:
    from jev_ml.domains.generic import adapter_for
    from jev_ml.domains.movie import INFO as MOVIE_INFO
    from jev_ml.domains.movie import available as movie_available

    base = _root(root)
    ok, reason = movie_available(base / "data" / "processed" if root is not None else None)
    items = [{**MOVIE_INFO.to_dict(), "available": ok, "reason": reason}]
    for p in sorted((base / "configs" / "domains").glob("*.yaml")):
        try:
            ad = adapter_for(p.stem, base)
        except Exception as exc:  # a broken config is listed, never raised
            items.append(
                {
                    "key": f"generic:{p.stem}",
                    "name": p.stem,
                    "description": "",
                    "entity_types": [],
                    "frequency": None,
                    "sources": [],
                    "capabilities": {},
                    "available": False,
                    "reason": f"invalid config {p.name}: {exc}",
                }
            )
            continue
        ok, reason = ad.available()
        items.append({**ad.info.to_dict(), "available": ok, "reason": reason})
    return items


def get_adapter(key: str, root: Path | None = None, **kwargs: Any) -> Any:
    """``movie`` -> ``MovieAdapter(**kwargs)``; ``generic:<name>`` (or ``<name>``) -> the generic adapter."""
    if key == "movie":
        from jev_ml.domains.movie import MovieAdapter

        return MovieAdapter(**kwargs)
    from jev_ml.domains.generic import adapter_for

    return adapter_for(key, _root(root))


__all__ = ["available", "get_adapter"]
