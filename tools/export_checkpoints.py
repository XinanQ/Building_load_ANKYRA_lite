"""Export the released carrier checkpoints: the stage-2 runs' final weights of the configuration a products mapping names.

Each research run directory of the products mapping (evaluation/products.json, ``run_dirs``) holds the weights file it
names (``final.pt``: {"model": state_dict, "epoch": 8}, the format ``ankyra_lite.load_carrier`` reads).  This tool copies
them byte for byte to

    <out>/<population>/<folder>/seed<s>.pt  population in {bdg2, cambridge, heew, eweld, goiener_confirm, households,
                                                          oslo, drammen, cineldi}; s = the mapping's seeds;
                                            folder = the mapping's ``checkpoint_folder`` ("" for recipe R, "rw" for
                                            recipe Rw; ankyra_lite._release.CHECKPOINT_SUBDIR reads the same folder)

checks every copy against the SHA-256 in its run record (COMPLETE.json), loads it with ``load_carrier`` (strict) and
counts its parameters, and writes ``<out>/MANIFEST.json``: population, seed, file, SHA-256, size, parameters, the
training cut (hour index of the population's record and its date), the store, the source run (relative to the research
root) with the SHA-256 of its records, and the from-scratch stage the run continued.  No absolute path is written.

Version 0.2.0 (a mapping with a ``configuration``, written by tools/products_for_configuration.py).  The files of the
0.1.x configuration (recipe R, seeds 0-2: <population>/seed[0-2].pt, byte-identical to 0.1.x) are exported as well, so
that the folder holds both; a file shared by both configurations is written once.  The manifest then has
``configurations`` ("release" and "0.1.x": recipe, seeds, folder pattern), ``default_configuration``, the ``use`` text for
three or six seeds, and per file ``recipe`` / ``recipe_key`` (R | Rw), ``seed_group`` (0 for seeds 0-2, 1 for 3-5),
``loss_weighting`` (null, or the frozen inverse-noise rule of recipe Rw), ``weights_record`` with its SHA-256,
``floor_sigma2`` and ``kish_ess_ratio_draws``, ``source_record`` (COMPLETE.json) and ``training_record`` (TRAINED.json)
with their SHA-256, ``start_checkpoint_sha256`` (STARTED.json) and ``in_release_configuration``.  The stage-2 epochs are
read from TRAINED.json (the study's COMPLETE.json of these runs keeps only the last epoch).

Verification (``--verify``, default oslo and cineldi): for a few test windows of each named population the exported
checkpoints forecast on the CPU (``ankyra_lite.carrier_forecasts`` at the origin and every pseudo-origin) and must
reproduce the run's stored carrier forecasts (carrier.npz, computed on a GPU) to a relative difference below 1e-4, at the
same issue times.  This reads the study's data stores through its own loaders (read-only); the carrier is given the
unit's load before the origin only (no target hour is passed).

The released files of version 0.1.0 (checkpoints/ of the repository, 27 files and MANIFEST.json) were written with
``--out checkpoints``; ``.gitignore`` ignores every other weights file (any ``*.pt`` but the released checkpoints of
the nine populations), so train your own carriers into ``runs/``.  Usage (CPU):

    python -B tools/export_checkpoints.py --out checkpoints                  # the released files of the repository
    python -B tools/export_checkpoints.py --out <dir> --verify oslo,cineldi --verify-windows 3
    python -B tools/export_checkpoints.py --products <mapping.json> --out <dir>     # another configuration

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
MAPPING_0_1 = REPO / "evaluation" / "products_0_1.json"             # the 0.1.x configuration (recipe R, seeds 0-2)
RECIPE_KEY = {"span return (R)": "R", "span return with inverse-noise loss weighting (Rw)": "Rw"}


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


def is_v020(prods):
    """True for a 0.2.0 mapping (written by tools/products_for_configuration.py)."""
    return bool(prods.get("configuration"))


def folder_of(prods):
    return str(prods.get("checkpoint_folder") or "")


def seeds_of(prods):
    return [int(s) for s in prods.get("seeds", (0, 1, 2))]


def stage2_history(rd, rec):
    """The stage-2 history of a run: TRAINED.json (study runs of 0.2.0) or COMPLETE.json (0.1.0 runs)."""
    tr = Path(rd) / "TRAINED.json"
    if tr.exists():
        h = _json(tr).get("history")
        if h:
            return h
    return rec.get("history") or ([rec["history_last"]] if rec.get("history_last") else [])


def loss_weighting(key):
    """The frozen inverse-noise rule of recipe Rw (ankyra_lite.carrier.train.NoiseWeights), or None for recipe R."""
    if key != "Rw":
        return None
    from dataclasses import asdict
    from ankyra_lite.carrier import train as TR
    nw = asdict(TR.NoiseWeights())
    return {"rule": "inverse-noise (frozen)", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in nw.items()}}


def export(out, prods, faces=ORDER, log=print, release=True, written=None):
    """Copy, check and describe every checkpoint of the mapping's configuration; returns the manifest entries.
    ``written``: {file: entry} of files already exported by another configuration (shared files are written once)."""
    from ankyra_lite import load_carrier
    from ankyra_lite.carrier.model import CARRIER_PARAMETERS, parameter_count
    v020 = is_v020(prods) or not release                        # the 0.2.0 fields, also on the 0.1.x files exported beside it
    folder = folder_of(prods)
    key = (prods.get("configuration_parts") or {}).get("recipe") or RECIPE_KEY.get(prods.get("recipe"), "R")
    seeds = seeds_of(prods)
    files = []
    for face in faces:
        e = prods["faces"][face]
        prep = _json(e["prep_path"])
        c = prep["cuts"]
        span = _json(e["span_path"]) if e.get("span_path") and Path(e["span_path"]).exists() else {}
        wrec = _json(CB.AO.parent / e["weights_record"]) if e.get("weights_record") else None
        for s, rd in zip(seeds, e["run_paths"]):
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
            rel_file = "/".join(x for x in (face, folder, f"seed{s}.pt") if x)
            dst = Path(out) / rel_file
            if written is not None and rel_file in written:
                if written[rel_file]["sha256"] != h:
                    raise SystemExit(f"{rel_file}: two configurations name different weights for the same file")
                written[rel_file]["configurations"].append("release" if release else "0.1.x")
                written[rel_file]["in_release_configuration"] |= bool(release)
                log(f"shared {rel_file}: already exported")
                continue
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
            hist = stage2_history(rd, rec)
            entry = {"population": face, "population_name": CB.SHOW_NAMES[face], "seed": s, "file": rel_file, "sha256": h,
                     "bytes": dst.stat().st_size, "parameters": n, "load": "ankyra_lite.load_carrier(path) (strict)",
                     "training_cut_hour": cut, "training_cut_date": (start0 + timedelta(hours=cut)).date().isoformat(),
                     "record_hour_0": start0.isoformat(), "store": e["store"],
                     "store_note": ("" if e["store"] == "original" else
                                    "CINELDI record of the corrected store (hour 0 = 2019-01-01; the original store's hour 0, truly 2019-03-01, is hour 1,416 here)"
                                    if face == "cineldi" else "trained with corrected CINELDI windows in its pooled training set"),
                     "recipe": prods.get("recipe"), "continuation_epochs": max([int(x["epoch"]) for x in hist] or [0]),
                     "returned_validation_windows": span.get("span_rows_target"),
                     "source_run": rel_to_research(rd), "source_record": "COMPLETE.json", "source_record_sha256": sha256(rec_f),
                     "source_record_utc": rec.get("utc")}
            st = _json(rd / "STARTED.json") if (rd / "STARTED.json").exists() else {}
            if v020:
                tr_f = rd / "TRAINED.json"
                entry.update({"recipe_key": key, "seed_group": int(s) // 3, "loss_weighting": loss_weighting(key),
                              "training_record": "TRAINED.json" if tr_f.exists() else None,
                              "training_record_sha256": sha256(tr_f) if tr_f.exists() else None,
                              "start_checkpoint_sha256": st.get("start_checkpoint_sha256"),
                              "configurations": ["release" if release else "0.1.x"], "in_release_configuration": bool(release)})
                if wrec is not None:
                    if st.get("weights_sha256") and st["weights_sha256"] != wrec.get("weights_sha256"):
                        raise SystemExit(f"{face} seed {s}: trained with other weights than the mapping's weights record")
                    entry.update({"weights_record": e["weights_record"], "weights_record_sha256": sha256(CB.AO.parent / e["weights_record"]),
                                  "floor_sigma2": wrec.get("floor_sigma2"), "kish_ess_ratio_draws": wrec.get("kish_ess_ratio_draws")})
                else:
                    entry.update({"weights_record": None})
            if e.get("start_paths"):
                sd = Path(e["start_paths"][seeds.index(s)])
                tr = _json(sd / "TRAINED.json") if (sd / "TRAINED.json").exists() else _json(sd / "COMPLETE.json")
                if st.get("start_checkpoint_sha256") and st["start_checkpoint_sha256"] != tr.get("checkpoint_sha256"):
                    raise SystemExit(f"{face} seed {s}: the span return did not start from the recorded from-scratch weights")
                entry.update({"from_scratch_run": rel_to_research(sd), "from_scratch_checkpoint_sha256": tr.get("checkpoint_sha256"),
                              "from_scratch_best_epoch": tr.get("best_epoch"), "from_scratch_epochs_run": tr.get("epochs_run")})
            files.append(entry)
            if written is not None:
                written[rel_file] = entry
            log(f"exported {face} seed {s} ({rel_file}): {n} parameters, cut {cut} ({entry['training_cut_date']}), sha256 {h[:12]}")
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
    import lite_reanchor as LR
    return LR.store_arrays(face)                                      # the store files alone (no window arrays, no results file)


def _row(a, b):
    a = np.asarray(a)
    return a[b] if a.ndim == 2 else a


def verify(out, prods, faces=VERIFY, windows=3, log=print):
    """Reproduce the stored carrier forecasts with the exported checkpoints on a few test windows (CPU).  The carrier
    gets the unit's load before the origin only (load[b, :o]); temperature and day types are the unit's rows."""
    import torch
    torch.set_num_threads(1)
    from ankyra_lite import carrier_forecasts, load_carrier
    folder = folder_of(prods)
    seeds = seeds_of(prods)
    res = {}
    for face in faces:
        e = prods["faces"][face]
        CB.wait_for_memory(f"verify {face}")
        t0 = time.time()
        d = _data(face, e)
        import lite_reanchor as LR
        qs = [np.load(Path(rd) / "carrier.npz", allow_pickle=False) for rd in e["run_paths"]]
        d["load"] = LR.history_only(np.asarray(d["load"], np.float64), np.concatenate([q["building"] for q in qs]),
                                    np.concatenate([q["origin"] for q in qs]))   # no hour at or after a unit's last test origin stays
        del qs
        per = []
        for s, rd in zip(seeds, e["run_paths"]):
            q = np.load(Path(rd) / "carrier.npz", allow_pickle=False)
            cut = int(q["cut"])
            pick = np.unique(np.linspace(0, len(q["building"]) - 1, windows).round().astype(int))
            model = load_carrier(Path(out) / face / folder / f"seed{s}.pt")
            worst, same = 0.0, True
            for i in pick:
                b, o = int(q["building"][i]), int(q["origin"][i])
                F = carrier_forecasts(model, np.asarray(d["load"][b, :o]), _row(d["temp"], b), _row(d["types"], b), d["groups"][b], d["start_iso"],
                                      unit=0, origin=o, cut=cut)
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


def _use(prods):
    n = len(seeds_of(prods))
    words = {3: "three", 6: "six"}.get(n, str(n))
    return ("one checkpoint per population and seed; pass the population's training cut as cut= and use origins at least 186 days after it; "
            f"a forecast with all {words} seeds is the plain average of their forecasts at every issue time, summed in ascending seed order "
            "(ankyra_lite.checkpoint_paths returns them so)" + ("; six seeds take about twice the carrier inference time of three" if n == 6 else ""))


def manifest_v020(prods, files, ver, t0, legacy=None):
    folder = folder_of(prods)
    key = prods["configuration_parts"]["recipe"]
    seeds = seeds_of(prods)
    configs = {"release": {"configuration": prods["configuration"], "recipe": prods.get("recipe"), "recipe_key": key, "seeds": seeds,
                           "continuation_layer": bool((prods.get("continuation") or {}).get("enabled")),
                           "files": "<population>/" + (f"{folder}/" if folder else "") + f"seed[{seeds[0]}-{seeds[-1]}].pt"}}
    if legacy is not None:
        configs["0.1.x"] = {"configuration": "R3", "recipe": legacy.get("recipe"), "recipe_key": "R", "seeds": seeds_of(legacy),
                            "continuation_layer": False, "files": "<population>/seed[0-2].pt"}
    return {"package": prods.get("label"), "version": prods.get("version"), "recipe": prods.get("recipe"),
            "recipe_description": prods.get("recipe_description"), "configurations": configs, "default_configuration": "release",
            "format": "torch.save of {'model': state_dict, 'epoch': 8} (the stage-2 run's final weights, copied byte for byte); "
                      "ankyra_lite.load_carrier(path) loads it strictly on the CPU",
            "use": _use(prods), "files": files, "count": len(files), "verification": ver,
            "verification_rule": f"stored carrier forecasts (carrier.npz, GPU) reproduced on the CPU to a relative difference below {REL_TOL:g}, "
                                 "same issue times, with every seed of the release configuration",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory, e.g. checkpoints (inside the repository only when publishing)")
    ap.add_argument("--recipe", default="R", choices=sorted(CB.RECIPE_PRODUCTS), help="products mapping (R: evaluation/products.json)")
    ap.add_argument("--products", default=None, help="another products mapping (overrides --recipe)")
    ap.add_argument("--faces", default=",".join(ORDER))
    ap.add_argument("--verify", default=",".join(VERIFY), help="populations whose stored carrier forecasts are reproduced ('' = none)")
    ap.add_argument("--verify-windows", type=int, default=3)
    ap.add_argument("--no-0-1", action="store_true", help="0.2.0 mapping: do not export the 0.1.x configuration's files as well")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if CB._in_read_only(out):
        raise SystemExit("the output directory lies in a read-only tree")
    prods = CB.load_products(Path(a.products) if a.products else CB.RECIPE_PRODUCTS[a.recipe])
    faces = [f for f in a.faces.split(",") if f]
    vf = [f for f in a.verify.split(",") if f]
    if any(f not in faces for f in vf):
        raise SystemExit("--verify names a population that is not exported")
    t0 = time.time()
    if is_v020(prods):
        written = {}
        files = export(out, prods, faces, release=True, written=written)
        legacy = None
        if not a.no_0_1:
            legacy = CB.load_products(MAPPING_0_1)
            files += export(out, legacy, faces, release=False, written=written)
        ver = verify(out, prods, vf, a.verify_windows) if vf else {}
        man = manifest_v020(prods, files, ver, t0, legacy)
    else:
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
