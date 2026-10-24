"""Rechnet das Raster der Messreihen vor (E1, E3 bis E6) und schreibt data/rank_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse (rund 15 Minuten)
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht speichern)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probezelle nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rkr_constants as C  # noqa: E402
import rkr_evaluation as ev  # noqa: E402

JOBS = {"rel": ev.relations_cell, "lever": ev.lever_cell, "size": ev.size_cell, "ties": ev.ties_cell, "cover": ev.cover_cell, "eff": ev.eff_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = []
    cover_ns = (10, 20) if quick else C.COVER_NS
    tasks += [("cover", (f, n, rho)) for n in sorted(cover_ns, reverse=True) for f in C.COVER_FAMILIES for rho in C.COVER_RHOS]       # die teuren Zellen zuerst
    tasks += [("rel", (f, rho)) for f in C.REL_FAMILIES for rho in C.REL_RHOS]
    tasks += [("lever", (n, k)) for n in sorted(C.LEVER_NS, reverse=True) for k in C.LEVER_KS]
    tasks += [("size", (f, n)) for n in sorted(C.SIZE_NS, reverse=True) for f in C.SIZE_FAMILIES]
    tasks += [("ties", (lv,)) for lv in C.TIE_LEVELS]
    tasks += [("eff", (f, rho)) for f in C.EFF_FAMILIES for rho in C.EFF_RHOS]
    if quick:
        tasks = [t for t in tasks if t[0] in ("cover", "ties", "eff")][:12] + [("lever", (10, 5)), ("size", ("t3", 30)), ("rel", ("normal", 0.6))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=14) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "boot_b": C.COVER_BOOT, "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("rank_grid_quick.json" if quick else "rank_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
