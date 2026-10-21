"""lite_reanchor.store_arrays against the study's store loader (tools_smallnet_carrier_v1.raw).

store_arrays opens the store files only.  For some populations the study's loader also opens the earlier study's window
arrays (with their targets and losses), its saved forecasts and its results file; the evaluation scripts that read no
target (re-anchoring, checkpoint export, cost timing) use store_arrays instead.  These tests check, population by
population, that both give the same arrays (load, temperature, day types bit for bit; names, groups, start timestamp
equal), and that store_arrays opens no file other than the store files.

The tests need the research folder (ANKYRA_LITE_RESEARCH_ROOT, default <repo>/../HKAD_V3) and its stores; each
population is skipped, with the reason, when the folder is absent or less memory is available than its arrays need
(ANKYRA_LITE_TEST_MIN_FREE_GB, default 5, plus the population's size).  ANKYRA_LITE_TEST_STORE_FACES (comma list)
narrows the populations.  Run from the repository root:  python -B -m unittest tests.test_store_arrays
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gc                                             # noqa: E402
import hashlib                                        # noqa: E402
import os                                             # noqa: E402
import unittest                                       # noqa: E402
from pathlib import Path                              # noqa: E402

import numpy as np                                    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "evaluation"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

NINE = ("bdg2", "cambridge", "heew", "eweld", "goiener_confirm", "households", "oslo", "drammen", "cineldi")
NEED_GB = {"households": 6.0, "eweld": 1.5}           # extra memory of the largest stores (one copy at a time)
MIN_FREE_GB = float(os.environ.get("ANKYRA_LITE_TEST_MIN_FREE_GB", "5"))
FACES = tuple(f for f in os.environ.get("ANKYRA_LITE_TEST_STORE_FACES", ",".join(NINE)).split(",") if f)


def _free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        return float("inf")


def _digest(d):
    """SHA-256 of every array and list the evaluation scripts read, with dtype and shape."""
    out = {}
    for k in ("load", "temp", "types"):
        a = np.ascontiguousarray(d[k])
        out[k] = (str(a.dtype), a.shape, hashlib.sha256(a.view(np.uint8).ravel()).hexdigest())
    for k in ("names", "groups"):
        out[k] = hashlib.sha256("\n".join(map(str, d[k])).encode("utf-8")).hexdigest()
    out["start_iso"] = str(d["start_iso"])
    return out


def _research_ok():
    try:
        import compare_baselines as CB
    except Exception:                                 # noqa: BLE001
        return None
    return CB if (CB.RESEARCH / "tools_smallnet_carrier_v1.py").exists() else None


class StoreArrays(unittest.TestCase):
    """store_arrays equals the study's loader on each population."""

    def _check(self, face):
        CB = _research_ok()
        if CB is None:
            self.skipTest("research folder absent")
        need = MIN_FREE_GB + NEED_GB.get(face, 0.0)
        if _free_gb() < need:
            self.skipTest(f"less than {need} GB available")
        import lite_reanchor as LR
        import warnings
        LR._paths()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tools_smallnet_carrier_v1                  # noqa: F401  (module imports are not counted)
            import tools_hcr_within_anchor_r3_v1              # noqa: F401
        mark = len(_OPENED)
        d = LR.store_arrays(face)
        files = [p for p in _OPENED[mark:] if not p.endswith((".py", ".pyc", ".pyd", ".dll")) and "site-packages" not in p]
        a = _digest(d)
        del d
        gc.collect()
        import tools_smallnet_carrier_v1 as V
        d = V.raw(face)
        b = _digest(d)
        del d
        gc.collect()
        self.assertEqual(a, b, f"{face}: store_arrays differs from the study's loader")
        data = [p for p in files if p.endswith((".npz", ".npy", ".json", ".csv", ".pt"))]
        self.assertTrue(data and all(p.endswith("store.npz") for p in data), f"{face}: store_arrays opened {data}")


_OPENED = []                                          # every file opened in this process (audit hook)


def _global_hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.fsdecode(args[0]).replace("\\", "/"))


sys.addaudithook(_global_hook)


def _make(face):
    def t(self):
        self._check(face)
    t.__doc__ = f"{face}: same arrays as tools_smallnet_carrier_v1.raw, store files only"
    return t


for _f in NINE:
    if _f in FACES:
        setattr(StoreArrays, f"test_{_f}", _make(_f))


if __name__ == "__main__":
    unittest.main()
