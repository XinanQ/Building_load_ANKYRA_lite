"""Export the released carrier checkpoints: the span-return runs' final weights, 9 populations x 3 seeds.

Each research run directory of the products mapping (evaluation/products.json, ``run_dirs``) holds the weights file it
names (``final.pt``: {"model": state_dict, "epoch": 8}, the format ``ankyra_lite.load_carrier`` reads).  This tool copies
them byte for byte to

    <out>/<population>/seed<s>.pt          population in {bdg2, cambridge, heew, eweld, goiener_confirm, households,
                                                          oslo, drammen, cineldi}; s in {0, 1, 2}

checks every copy against the SHA-256 in its run record (COMPLETE.json), loads it with ``load_carrier`` (strict) and
counts its parameters, and writes ``<out>/MANIFEST.json``: population, seed, file, SHA-256, size, parameters, the
training cut (hour index of the population's record and its date), the store, the source run (relative to the research
root) with the SHA-256 of its record, and the from-scratch stage the span return continued.  No absolute path is
written.

Verification (``--verify``, default oslo and cineldi): for a few test windows of each named population the exported
checkpoints forecast on the CPU (``ankyra_lite.carrier_forecasts`` at the origin and every pseudo-origin) and must
reproduce the run's stored carrier forecasts (carrier.npz, computed on a GPU) to a relative difference below 1e-4, at the
same issue times.  This reads the study's data stores through its own loaders (read-only).

The released files of version 0.1.0 (checkpoints/ of the repository, 27 files and MANIFEST.json) were written with
``--out checkpoints``; ``.gitignore`` ignores every other weights file (any ``*.pt`` but checkpoints/<population>/seed[0-2].pt
of the nine populations), so train your own carriers into ``runs/``.  Usage (CPU):

    python -B tools/export_checkpoints.py --out checkpoints                  # the released files of the repository
    python -B tools/export_checkpoints.py --out <dir> --verify oslo,cineldi --verify-windows 3

Environment: ANKYRA_LITE_RESEARCH_ROOT (default <repo>/../HKAD_V3), ANKYRA_ORIGINAL_ROOT (default
<repo>/../Building_load_ANKYRA); both are read-only.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse                                                    # noqa: E402
import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import shutil                                                      # noqa: E402
import time                                                        # noqa: E402
import warnings                                                    # noqa: E402
from datetime import datetime, timedelta                           # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "evaluation"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
import compare_baselines as CB                                     # noqa: E402  (paths, products mapping, read-only guard)

ORDER = ("bdg2", "cambridge", "heew", "eweld", "goiener_confirm", "households", "oslo", "drammen", "cineldi")
VERIFY = ("oslo", "cineldi")
REL_TOL = 1e-4                                                      # CPU against the stored GPU forecasts (float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel_to_research(p):
    """A research path as it is written in the manifest: relative to the research root, forward slashes."""
    return Path(p).resolve().relative_to(CB.RESEARCH).as_posix()


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def export(out, prods, faces=ORDER, log=print):
    """Copy, check and describe every checkpoint; returns the manifest entries."""
    from ankyra_lite import load_carrier
    from ankyra_lite.carrier.model import CARRIER_PARAMETERS, parameter_count
    files = []
    for face in faces:
        e = prods["faces"][face]
        prep = _json(e["prep_path"])
        c = prep["cuts"]
        span = _json(e["span_path"]) if e.get("span_path") and Path(e["span_path"]).exists() else {}
        for s, rd in enumerate(e["run_paths"]):
            rd = Path(rd)
            src = rd / e["checkpoint"]
            rec_f = rd / "COMPLETE.json"
            rec = _json(rec_f)
            h = sha256(src)
            if h != rec.get("checkpoint_sha256"):
                raise SystemExit(f"{face} seed {s}: the checkpoint differs from its run record")
            cut = int(np.load(rd / "carrier.npz", allow_pickle=False)["cut"])
            if cut != int(c["Cprime"]):
                raise SystemExit(f"{face} seed {s}: the run's training cut {cut} differs from the split record {c['Cprime']}")
            start0 = datetime.fromisoformat(c["Cprime_label"]) - timedelta(hours=int(c["Cprime"]))
            dst = Path(out) / face / f"seed{s}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            if sha256(dst) != h:
                raise SystemExit(f"{dst}: copy differs from the source")
            model = load_carrier(dst)                                              # strict state-dict load, CPU, eval mode
            n = parameter_count(model)
            if n != CARRIER_PARAMETERS:
                raise SystemExit(f"{face} seed {s}: {n} parameters, expected {CARRIER_PARAMETERS}")
            entry = {"population": face, "population_name": CB.SHOW_NAMES[face], "seed": s, "file": f"{face}/seed{s}.pt", "sha256": h,
                     "bytes": dst.stat().st_size, "parameters": n, "load": "ankyra_lite.load_carrier(path) (strict)",
                     "training_cut_hour": cut, "training_cut_date": (start0 + timedelta(hours=cut)).date().isoformat(),
                     "record_hour_0": start0.isoformat(), "store": e["store"],
                     "store_note": ("" if e["store"] == "original" else
                                    "CINELDI record of the corrected store (hour 0 = 2019-01-01; the original store's hour 0, truly 2019-03-01, is hour 1,416 here)"
                                    if face == "cineldi" else "trained with corrected CINELDI windows in its pooled training set"),
                     "recipe": prods.get("recipe"), "continuation_epochs": max(int(x["epoch"]) for x in rec.get("history", [{"epoch": 0}])),
                     "returned_validation_windows": span.get("span_rows_target"),
                     "source_run": rel_to_research(rd), "source_record": "COMPLETE.json", "source_record_sha256": sha256(rec_f),
                     "source_record_utc": rec.get("utc")}
            if e.get("start_paths"):
                sd = Path(e["start_paths"][s])
                tr = _json(sd / "TRAINED.json") if (sd / "TRAINED.json").exists() else _json(sd / "COMPLETE.json")
                st = _json(rd / "STARTED.json") if (rd / "STARTED.json").exists() else {}
                if st.get("start_checkpoint_sha256") and st["start_checkpoint_sha256"] != tr.get("checkpoint_sha256"):
                    raise SystemExit(f"{face} seed {s}: the span return did not start from the recorded from-scratch weights")
                entry.update({"from_scratch_run": rel_to_research(sd), "from_scratch_checkpoint_sha256": tr.get("checkpoint_sha256"),
                              "from_scratch_best_epoch": tr.get("best_epoch"), "from_scratch_epochs_run": tr.get("epochs_run")})
            files.append(entry)
            log(f"exported {face} seed {s}: {n} parameters, cut {cut} ({entry['training_cut_date']}), sha256 {h[:12]}")
    return files


def _data(face, entry):
    """The population's store arrays through the study's own loader (the corrected CINELDI store for a corrected-store
    CINELDI product)."""
    for p in (str(CB.REL), str(CB.HKAD)):
        if p not in sys.path:
            sys.path.insert(0, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if CB.on_corrected_cineldi(face, entry):
            import tools_smallnet_sr_formal_dc_v1 as DC                       # installs the corrected-store loader for CINELDI
            return DC.V.raw(face)
        import tools_smallnet_carrier_v1 as V
        return V.raw(face)


def verify(out, prods, faces=VERIFY, windows=3, log=print):
    """Reproduce the stored carrier forecasts with the exported checkpoints on a few test windows (CPU)."""
    import torch
    torch.set_num_threads(1)
    from ankyra_lite import carrier_forecasts, load_carrier
    res = {}
    for face in faces:
        e = prods["faces"][face]
        CB.wait_for_memory(f"verify {face}")
        t0 = time.time()
        d = _data(face, e)
        per = []
        for s, rd in enumerate(e["run_paths"]):
            q = np.load(Path(rd) / "carrier.npz", allow_pickle=False)
            cut = int(q["cut"])
            pick = np.unique(np.linspace(0, len(q["building"]) - 1, windows).round().astype(int))
            model = load_carrier(Path(out) / face / f"seed{s}.pt")
            worst, same = 0.0, True
            for i in pick:
                b, o = int(q["building"][i]), int(q["origin"][i])
                F = carrier_forecasts(model, d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], unit=b, origin=o, cut=cut)
                ks = [k for k in range(q["have"].shape[1]) if q["have"][i, k]]
                same &= sorted(F) == ks
                for k in F:
                    ref = q["pred"][i, k].astype(np.float64)
                    worst = max(worst, float(np.abs(F[k] - ref).max() / max(np.abs(ref).mean(), 1e-9)))
            per.append({"seed": s, "windows": [int(x) for x in pick], "max_rel_diff": worst, "same_issue_times": bool(same)})
            log(f"verify {face} seed {s}: windows {list(map(int, pick))}, largest relative difference {worst:.2e}, same issue times {same}")
        res[face] = {"store": e["store"], "seeds": per, "pass": all(x["max_rel_diff"] < REL_TOL and x["same_issue_times"] for x in per),
                     "seconds": round(time.time() - t0, 1)}
        del d
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory, e.g. checkpoints (inside the repository only when publishing)")
    ap.add_argument("--recipe", default="R", choices=sorted(CB.RECIPE_PRODUCTS), help="products mapping (R: the released recipe)")
    ap.add_argument("--faces", default=",".join(ORDER))
    ap.add_argument("--verify", default=",".join(VERIFY), help="populations whose stored carrier forecasts are reproduced ('' = none)")
    ap.add_argument("--verify-windows", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if CB._in_read_only(out):
        raise SystemExit("the output directory lies in a read-only tree")
    prods = CB.load_products(CB.RECIPE_PRODUCTS[a.recipe])
    faces = [f for f in a.faces.split(",") if f]
    vf = [f for f in a.verify.split(",") if f]
    if any(f not in faces for f in vf):
        raise SystemExit("--verify names a population that is not exported")
    t0 = time.time()
    files = export(out, prods, faces)
    ver = verify(out, prods, vf, a.verify_windows) if vf else {}
    man = {"package": prods.get("label"), "version": prods.get("version"), "recipe": prods.get("recipe"),
           "recipe_description": prods.get("recipe_description"),
           "format": "torch.save of {'model': state_dict, 'epoch': 8} (the span-return run's final weights, copied byte for byte); "
                     "ankyra_lite.load_carrier(path) loads it strictly on the CPU",
           "use": "one checkpoint per population and seed; pass the population's training cut as cut= and use origins at least 186 days after it; "
                  "a forecast with all three seeds is the plain average of their forecasts",
           "files": files, "count": len(files), "verification": ver,
           "verification_rule": f"stored carrier forecasts (carrier.npz, GPU) reproduced on the CPU to a relative difference below {REL_TOL:g}, same issue times",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
    text = CB.scrub(json.dumps(man, indent=1, ensure_ascii=False))
    (out / "MANIFEST.json").write_bytes(text.encode("utf-8"))
    bad = [f for f, v in ver.items() if not v["pass"]]
    print(f"{len(files)} checkpoints exported to {out.name}/; MANIFEST.json written; verification "
          + ("passed" if ver and not bad else ("not run" if not ver else f"FAILED on {bad}")))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
