"""Phase 1 smoke test for the 24-shot PatchCore predictor (MVTec-AD).

Validates, for a single category and a small number of test images, that:

  1. PatchCore builds the memory bank from exactly ``--memory-size`` normal
     training images (fixed seed, paths recorded);
  2. query inference runs and yields a finite image-level anomaly score;
  3. anomaly-map files really exist on disk;
  4. the memory bank contains **no** test image (no leakage);
  5. a second run reuses the on-disk memory cache instead of re-fitting.

Usage (from anywhere; the predictor inserts the vendored anomalib/src into
sys.path itself)::

    /user/pfy/anaconda3/envs/anomalyagent/bin/python smoke_test_phase1.py \
        --category bottle --memory-size 24 --seed 0

Run twice to see the cache MISS on the first invocation and HIT on the second,
or use ``--fresh`` to wipe the category cache before the first run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from this TEST/ folder (or anywhere) while the implementation
# module lives at the repo root (one level up).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from patchcore_predictor import PatchcorePredictor

DEFAULT_DATA_ROOT = "/data/pfy/dataset/MVTec-AD"
DEFAULT_CACHE_ROOT = "/data/pfy/AgentIAD/results/phase1/cache"
DEFAULT_OUTPUT_ROOT = "/data/pfy/AgentIAD/results/phase1/out"


def discover_query_images(data_root: Path, category: str) -> list[Path]:
    """Pick one normal test image and the first available defective test image."""
    test_dir = data_root / category / "test"
    queries: list[Path] = []
    good_dir = test_dir / "good"
    good = sorted(good_dir.glob("*.png")) if good_dir.is_dir() else []
    if good:
        queries.append(good[0])
    defect_types = sorted(p.name for p in test_dir.iterdir() if p.is_dir() and p.name != "good")
    for dtype in defect_types:
        files = sorted((test_dir / dtype).glob("*.png"))
        if files:
            queries.append(files[0])
            break
    if not queries:
        raise RuntimeError(f"[{category}] no test images found under {test_dir}")
    return queries


def collect_test_images(data_root: Path, category: str) -> list[Path]:
    """All absolute paths that live under ``<category>/test`` (leakage scan)."""
    test_dir = data_root / category / "test"
    return sorted(p.resolve() for p in test_dir.rglob("*.png"))


def check(memory_meta: dict, queries: list[Path],
          predictor: PatchcorePredictor) -> dict:
    """Run predictions and assert the Phase-1 acceptance criteria."""
    checks: dict = {}
    category = memory_meta["category"]

    # --- 1. memory built, on disk, from exactly memory_size normal images ----
    mem_path = predictor.cache_root / category / "memory.pt"
    checks["memory_built"] = bool(memory_meta.get("n_memory_patches", 0) > 0)
    checks["memory_file_exists"] = mem_path.is_file()
    checks["n_memory_images"] = len(memory_meta["memory_images"])
    checks["n_memory_images_ok"] = checks["n_memory_images"] == memory_meta["memory_size"]
    checks["memory_patches"] = memory_meta.get("n_memory_patches")

    # --- 5. no test leakage ------------------------------------------------
    mem_abs = {Path(p) for p in memory_meta["memory_images"]}
    test_abs = set(collect_test_images(predictor.data_root, category))
    checks["memory_all_from_train_good"] = all(
        Path(p).is_relative_to(predictor.data_root.resolve() / category / "train" / "good")
        for p in mem_abs
    )
    checks["memory_disjoint_from_test"] = len(mem_abs & test_abs) == 0
    checks["n_test_images_scanned"] = len(test_abs)

    # --- per-query inference checks -----------------------------------------
    checks["queries"] = []
    ok = True
    for q in queries:
        rec = predictor.predict(q, category)
        finite = isinstance(rec["image_score"], float) and __import__("math").isfinite(rec["image_score"])
        map_ok = Path(rec["anomaly_map_path"]).is_file() and Path(rec["anomaly_map_path"]).stat().st_size > 0
        npy_ok = Path(rec["anomaly_map_raw_path"]).is_file()
        checks["queries"].append({
            "query": rec["query_image"],
            "image_score": rec["image_score"],
            "score_finite": finite,
            "anomaly_map_path": rec["anomaly_map_path"],
            "map_png_exists": map_ok,
            "map_raw_exists": npy_ok,
        })
        ok = ok and finite and map_ok and npy_ok

    checks["query_inference_ok"] = ok
    return checks


def run(args) -> dict:
    """One predictor run: build-or-load memory, then predict over queries."""
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    output_root = Path(args.output_root)
    category = args.category

    predictor = PatchcorePredictor(
        data_root=data_root,
        cache_root=cache_root,
        output_root=output_root,
        seed=args.seed,
        memory_size=args.memory_size,
        device=args.device,
    )

    queries = [Path(q) for q in args.query_images] if args.query_images else discover_query_images(data_root, category)

    print(f"\n=== {category}: queries = {queries} ===", flush=True)
    memory_meta = predictor.get_memory(category)
    print(f"[{category}] loaded_from_cache = {memory_meta.get('loaded_from_cache')}")

    checks = check(memory_meta, queries, predictor)

    # Record the 24 used normal-image paths next to the predictions.
    record = {
        "dataset": memory_meta["dataset"],
        "category": memory_meta["category"],
        "memory_size": memory_meta["memory_size"],
        "seed": memory_meta["seed"],
        "category_seed": memory_meta["category_seed"],
        "memory_images": memory_meta["memory_images"],
        "loaded_from_cache": memory_meta["loaded_from_cache"],
        "checks": checks,
    }
    out_dir = output_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "memory_images_24.json"
    with open(list_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"24-image list + checks written to {list_path}", flush=True)

    mtime = (cache_root / category / "memory.pt").stat().st_mtime_ns if (cache_root / category / "memory.pt").is_file() else None
    print(f"[{category}] checks = {json.dumps(checks, indent=2)}", flush=True)
    return {"predictor": predictor, "checks": checks, "memory_meta": memory_meta,
            "queries": queries, "memory_file_mtime_ns": mtime}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 PatchCore 24-shot smoke test")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--memory-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-images", nargs="*", default=None,
                        help="optional explicit query paths; default auto-picks good+defective test images")
    parser.add_argument("--device", default=None,
                        help="torch device, e.g. cuda:0 or cpu (default auto)")
    parser.add_argument("--fresh", action="store_true",
                        help="wipe the category cache before run 1 so the build path is exercised")
    args = parser.parse_args()

    # Silence noisy Anomalib/timm FutureWarnings.
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    category_cache = Path(args.cache_root) / args.category
    if args.fresh and category_cache.exists():
        import shutil
        shutil.rmtree(category_cache)
        print(f"wiped category cache: {category_cache}", flush=True)

    print("=" * 78)
    print(f"RUN 1  (expect: memory cache MISS -> fit + cache)")
    print("=" * 78)
    r1 = run(args)
    ok1 = all(r1["checks"][k] for k in
              ("memory_built", "memory_file_exists", "n_memory_images_ok",
               "memory_all_from_train_good", "memory_disjoint_from_test", "query_inference_ok"))
    print(f"RUN1 all-checks-ok = {ok1}", flush=True)

    print("=" * 78)
    print(f"RUN 2  (expect: memory cache HIT -> reuse, no re-fit)")
    print("=" * 78)
    r2 = run(args)
    ok2 = all(r2["checks"][k] for k in
              ("memory_built", "memory_file_exists", "n_memory_images_ok",
               "memory_all_from_train_good", "memory_disjoint_from_test", "query_inference_ok"))
    cache_reused = bool(r2["memory_meta"].get("loaded_from_cache") is True)
    mtime_unchanged = r1["memory_file_mtime_ns"] == r2["memory_file_mtime_ns"]
    same_memory_images = r1["memory_meta"]["memory_images"] == r2["memory_meta"]["memory_images"]
    print(f"RUN2 all-checks-ok = {ok2}", flush=True)
    print(f"cache reused on RUN2        = {cache_reused}", flush=True)
    print(f"memory.pt mtime unchanged   = {mtime_unchanged}", flush=True)
    print(f"same 24 memory images both runs = {same_memory_images}", flush=True)

    passed = ok1 and ok2 and cache_reused and mtime_unchanged and same_memory_images
    print("=" * 78)
    print("PHASE 1 SMOKE TEST", "PASSED" if passed else "FAILED")
    print("=" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
