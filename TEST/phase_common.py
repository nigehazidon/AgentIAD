"""Shared, dependency-free helpers for the phase drivers.

``list_entries`` and ``limit_round_robin`` are the same helpers the Phase-3
driver uses, lifted here rather than imported from ``run_phase3_observation``:
importing that module pulls in ``patchcore_regions`` -> ``patchcore_predictor``
-> the vendored Anomalib copy, which reorders ``sys.path``.  This module is pure
stdlib, so any driver can import it first, in any order.
"""

from __future__ import annotations

import json
from pathlib import Path


def list_entries(map_root: Path, category: str) -> list[dict]:
    """Every cached test image of ``category``, from the Phase-2 score files.

    The score files are the authoritative enumeration: each one carries the
    absolute ``query_image`` path, so no path is ever reconstructed from a stem.
    """
    cat_dir = map_root / category
    if not cat_dir.is_dir():
        raise RuntimeError(f"[{category}] no cached anomaly maps under {cat_dir}")

    entries: list[dict] = []
    for score_file in sorted(cat_dir.glob("*__score.json")):
        stem = score_file.name[: -len("__score.json")]
        map_path = cat_dir / f"{stem}__anomaly_map.npy"
        if not map_path.is_file():
            continue
        with open(score_file) as f:
            rec = json.load(f)
        entries.append({
            "stem": stem,
            "query_image": rec["query_image"],
            "image_score": rec.get("image_score"),
            "map_path": map_path,
            # authoritative, taken from the query path rather than the stem
            "defect_type": Path(rec["query_image"]).parent.name,
        })
    if not entries:
        raise RuntimeError(f"[{category}] no cached anomaly maps in {cat_dir}")
    return entries


def defect_types_of(entries: list[dict]) -> list[str]:
    """Sorted defect-type names present in ``entries``."""
    return sorted({e["defect_type"] for e in entries})


def limit_round_robin(entries: list[dict], n: int | None) -> list[dict]:
    """Take ``n`` entries spread across the defect types.

    Round-robin rather than a prefix, so a smoke run always includes the
    ``good`` (normal) test images and not only the first defect type.
    """
    if n is None or n >= len(entries):
        return entries
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["defect_type"], []).append(entry)

    out: list[dict] = []
    depth = 0
    while len(out) < n:
        added = False
        for name in sorted(groups):
            if depth < len(groups[name]):
                out.append(groups[name][depth])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        depth += 1
    return out
