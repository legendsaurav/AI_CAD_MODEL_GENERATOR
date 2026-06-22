"""
scripts/run_all.py  –  Master End-to-End Pipeline Runner (Version 0 → 5)
=========================================================================
Executes all pipeline stages in sequence and prints a final pass/fail report.
Run from the repo root:
    python scripts/run_all.py
"""
import os
import sys
import traceback
import time

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.config import ConfigManager


# ── Stage registry ────────────────────────────────────────────────────────

STAGES = [
    ("Version 0 – Hook Validation   (Unit Tests)",  "v0_hooks"),
    ("Version 1 – Temporal Probing",                "v1_probing"),
    ("Version 2 – Graph Extraction",               "v2_graph"),
    ("Version 3 – Primitive Recovery",             "v3_primitives"),
    ("Version 4+5 – GGL Export & CAD Macro",       "v4v5_cad"),
]

results = {}     # stage_key → ("PASS" | "FAIL", elapsed_s, error_msg)


# ── Stage runners ─────────────────────────────────────────────────────────

def run_v0_hooks():
    """Run unit tests using unittest runner (no pytest required)."""
    import unittest
    loader = unittest.TestLoader()
    suite  = loader.discover(
        start_dir=os.path.join(_REPO_ROOT, "tests"),
        pattern="test_*.py",
    )
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)
    if result.failures or result.errors:
        msgs = [str(f[1]) for f in result.failures + result.errors]
        raise AssertionError(f"{len(result.failures + result.errors)} test(s) failed:\n" + "\n".join(msgs[:3]))
    print(f"   ✅ {result.testsRun} tests passed.")


def run_v1_probing():
    from scripts.run_probing import main as v1
    ConfigManager.reset()
    rankings = v1()
    assert rankings and len(rankings) > 0, "No layer rankings returned"
    print(f"   ✅ Best layer: {rankings[0][0]}  (score={rankings[0][1]:.4f})")


def run_v2_graph():
    from scripts.run_graph_extraction import main as v2
    ConfigManager.reset()
    ggl = v2()
    assert ggl is not None, "No GGL returned"
    print(f"   ✅ Graph: {len(ggl.nodes)} nodes, {len(ggl.edges)} edges")


def run_v3_primitives():
    from scripts.run_primitive_recovery import main as v3
    ConfigManager.reset()
    ggl = v3()
    prims = [n for n in ggl.nodes if n.type in {"Cylinder", "Box", "Sphere", "Cone", "Plane"}]
    assert len(prims) > 0, "No primitives were added to the GGL"
    print(f"   ✅ {len(prims)} primitive(s) recovered")


def run_v4v5_cad():
    from scripts.run_ggl_export import main as v45
    ConfigManager.reset()
    ggl = v45()
    assert ggl is not None, "No GGL returned from V4/V5"
    prims = [n for n in ggl.nodes if n.type in {"Cylinder", "Box", "Sphere"}]
    print(f"   ✅ Validated & exported {len(prims)} primitive(s) to GGL JSON")


RUNNERS = {
    "v0_hooks":    run_v0_hooks,
    "v1_probing":  run_v1_probing,
    "v2_graph":    run_v2_graph,
    "v3_primitives": run_v3_primitives,
    "v4v5_cad":    run_v4v5_cad,
}


# ── Entry-point ───────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║   GEOMETRY ENGINE  –  Full Pipeline Run (V0 → V5)           ║")
    print("╚" + "═" * 62 + "╝\n")

    for label, key in STAGES:
        print(f"{'─' * 64}")
        print(f"▶  {label}")
        print(f"{'─' * 64}")
        t0 = time.time()
        try:
            RUNNERS[key]()
            elapsed = time.time() - t0
            results[key] = ("PASS", elapsed, "")
        except Exception as exc:
            elapsed = time.time() - t0
            results[key] = ("FAIL", elapsed, traceback.format_exc())
            print(f"   ❌ FAILED: {exc}")

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║   PIPELINE SUMMARY                                           ║")
    print("╠" + "═" * 62 + "╣")

    all_pass = True
    for label, key in STAGES:
        status, elapsed, err = results.get(key, ("SKIP", 0, ""))
        icon   = "✅" if status == "PASS" else "❌"
        suffix = f"{elapsed:.1f}s"
        short  = label[:45].ljust(45)
        print(f"║  {icon}  {short}  {suffix:>6}  ║")
        if status != "PASS":
            all_pass = False

    print("╠" + "═" * 62 + "╣")
    outcome = "ALL STAGES PASSED 🎉" if all_pass else "SOME STAGES FAILED – see above"
    print(f"║   {outcome.ljust(59)}║")
    print("╚" + "═" * 62 + "╝\n")

    if not all_pass:
        print("── Detailed error traces ──────────────────────────────────────")
        for label, key in STAGES:
            status, _, err = results.get(key, ("PASS", 0, ""))
            if status == "FAIL" and err:
                print(f"\n[{label}]\n{err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
