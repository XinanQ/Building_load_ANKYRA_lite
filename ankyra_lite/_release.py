"""Release switches of ankyra-lite: the only place where the delivered configuration is chosen.

  VERSION               the package version (``ankyra_lite.__version__``)
  CONTINUATION_DEFAULT  default of ``forecast(..., continuation=...)``: False (off) or True (the origin-continuity
                        layer with its frozen constants, ``ankyra_lite.continuation.CONSTANTS``)
  CARRIER_RECIPE        the recipe of the released default checkpoints: "R" or "Rw"
  CARRIER_SEEDS         the seeds of the default ensemble, ascending (their forecasts are averaged at every issue time)
  CHECKPOINT_SUBDIR     the folder of those checkpoints under checkpoints/<population>/ ("" for recipe R, "rw" for
                        recipe Rw)

Versions 0.2.0 and 0.2.1 set CONTINUATION_DEFAULT = True, CARRIER_RECIPE = "Rw", CARRIER_SEEDS = (0, 1, 2, 3, 4, 5) and
CHECKPOINT_SUBDIR = "rw".  With CONTINUATION_DEFAULT = False, CARRIER_RECIPE = "R", CARRIER_SEEDS = (0, 1, 2) and
CHECKPOINT_SUBDIR = "", the package is meant to reproduce ankyra-lite 0.1.1; this is checked bit for bit by the two examples
and the 0.1.1 comparison tests (tests/test_forecast_v2.py, tests/test_train_rw.py), not by a full re-run of every results
table.  ``forecast`` reads CONTINUATION_DEFAULT once, when ``ankyra_lite.forecast`` is imported.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["VERSION", "CONTINUATION_DEFAULT", "CARRIER_RECIPE", "CARRIER_SEEDS", "CHECKPOINT_SUBDIR", "checkpoint_paths"]

VERSION = "0.2.1"
CONTINUATION_DEFAULT = True
CARRIER_RECIPE = "Rw"
CARRIER_SEEDS = (0, 1, 2, 3, 4, 5)
CHECKPOINT_SUBDIR = "rw"


def checkpoint_paths(population: str, root=None) -> list[Path]:
    """The default checkpoints of a population in ascending seed order:
    <root>/<population>/<CHECKPOINT_SUBDIR>/seed<k>.pt for k in CARRIER_SEEDS (root: the repository's checkpoints/;
    an installed package holds no checkpoints, so pass ``root`` there).  Pass them to ``load_carrier`` and the
    resulting list to ``forecast`` (the ensemble average sums the members in the order given).  The examples read the
    same folder."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "checkpoints"
    return [base / population / CHECKPOINT_SUBDIR / f"seed{k}.pt" for k in sorted(CARRIER_SEEDS)]
