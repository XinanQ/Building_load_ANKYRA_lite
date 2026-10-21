"""Write the products mapping of one released configuration (evaluation/products.json or another file).

A configuration of version 0.2.0 is a carrier recipe, a number of seeds and the origin-continuity layer switched on or
off:

    --recipe R | Rw         recipe R (span return, the recipe of 0.1.x) or recipe Rw (the span return with the
                            inverse-noise weighted loss in stage 2)
    --seeds 3 | 6           seeds 0, 1, 2 or seeds 0-5, averaged at every issue time
    --continuation on | off the origin-continuity layer (ankyra_lite.continuation, frozen constants) on the anchored
                            trajectory, readout rule A (energy, level and peak read before the layer)
    --recipe r010           the mapping of the 0.1.x releases (evaluation/products_0_1.json, copied unchanged)

Every product of the 0.2.0 study is the output of the study's own anchoring step on the formal training cut:

    analysis_outputs/lite_upgrade_v2/formal/late/<population>/arms_<NAME>.npz       before the layer (the 0.1.0 format:
                                                                                     building, origin, traj, level, peak,
                                                                                     off, micro, state, cont_* inputs)
    analysis_outputs/lite_upgrade_v2/formal/late_cont/<population>/arms_<NAME>.npz  the same with the layer (traj = F',
                                                                                     traj_anchored = F, cont_* record)

with NAME = <recipe>_ens6 (six seeds), <recipe>_g0 (seeds 0-2) or <recipe>_g1 (seeds 3-5); R_g0 equals the 0.1.0
product bit for bit.  The mapping names, for each of the nine populations:

    npz / continued_npz     the product before and, when the layer is on, after the layer ("npz" keeps the 0.1.0
                            format, so every reader of 0.1.x keeps working)
    group_npz               the two three-seed groups (seeds 0-2, 3-5) of a six-seed configuration, before the layer;
    group_continued_npz     with the layer (the seed spread is read from them; no single-seed product exists)
    run_dirs                the stage-2 run of every seed (recipe R: the released 0.1.0 runs for seeds 0-2 and the
                            study's runs for seeds 3-5; recipe Rw: the study's runs Rw_s0..Rw_s5)
    start_run_dirs          the from-scratch stage each run continued (from its STARTED.json, hash-checked)
    span_plan, prep, store  as in 0.1.x (the returned validation span, temp_sigma_std and the split record)
    weights_record          recipe Rw: the record of the frozen inverse-noise weights of the stage-2 pool
    recipe_variants         the other configurations on the same windows (the 0.1.0 product, R and Rw with three and
                            six seeds, each with and without the layer) and the 0.1.x recipe arms P3s, F, V

Checks before anything is written (the mapping is not written when one fails): every path exists; each product's SHA-256
equals its .json record; the record names the configuration's recipe and seeds, and the SHA-256 of every seed's carrier
forecasts equals the run directory's carrier.npz; each run's checkpoint equals its COMPLETE.json; each run started from
the recorded from-scratch weights; recipe Rw runs used the recorded weights; each continued product was made from the
mapped product with the release's constants and rule (ankyra_lite.continuation.CONSTANTS, DIGEST); every product has
the windows and state flags of the 0.1.0 product; R_g0 reproduces the 0.1.0 product bit for bit.  Nothing reads a
target: the products hold forecasts, readouts, state flags and the layer's pre-origin inputs only.

Usage (from the repository root):

    python -B tools/products_for_configuration.py --recipe Rw --seeds 6 --continuation on --out evaluation/products.json
    python -B tools/products_for_configuration.py --recipe r010 --out evaluation/products.json      # back to 0.1.x

The default output is evaluation/products_v020_candidate.json; evaluation/products.json is written only when named
with --out.  Paths in the mapping are relative to the research root (ANKYRA_LITE_RESEARCH_ROOT, default
<repo>/../HKAD_V3), which is read-only here.
"""
from __future__ import annotations

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse                                                    # noqa: E402
import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import time                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO / "evaluation"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import compare_baselines as CB                                     # noqa: E402  (paths, read-only guard)

STUDY = "analysis_outputs/lite_upgrade_v2"                         # the 0.2.0 study, relative to the research root
NINE = CB.NINE
DEFAULT_OUT = REPO / "evaluation" / "products_v020_candidate.json"
MAPPING_0_1 = REPO / "evaluation" / "products_0_1.json"
LABEL, VERSION = "ankyra-lite 0.2.0", "0.2.0"
RECIPE_NAME = {"R": "span return (R)", "Rw": "span return with inverse-noise loss weighting (Rw)"}
GROUPS6 = ([0, 1, 2], [3, 4, 5])
OLD_ARMS = ("P3s", "F", "V")                                       # the 0.1.x recipe arms (pre-layer, three seeds)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    """A research path as written in the mapping: relative to the research root, forward slashes."""
    return Path(p).resolve().relative_to(CB.RESEARCH).as_posix()


def research(r) -> Path:
    return CB.RESEARCH / r


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def config_code(recipe, seeds, continuation):
    """The study's code of a configuration: R3, R6, Rw3, Rw6, with '_c' when the layer is on."""
    return f"{recipe}{seeds}" + ("_c" if continuation else "")


def product_name(recipe, seeds):
    """The study's product name: <recipe>_ens6 for six seeds, <recipe>_g0 (seeds 0-2) for three."""
    return f"{recipe}_ens6" if seeds == 6 else f"{recipe}_g0"


def late(face, name):
    return f"{STUDY}/formal/late/{face}/arms_{name}.npz"


def late_cont(face, name):
    return f"{STUDY}/formal/late_cont/{face}/arms_{name}.npz"


def _tail(path_text):
    """The part of a recorded absolute path from 'analysis_outputs' on (records may carry another drive or user)."""
    s = str(path_text).replace("\\", "/")
    i = s.find("analysis_outputs/")
    return s[i:] if i >= 0 else None


def describe(recipe, seeds, continuation):
    words = {3: "seeds 0, 1 and 2", 6: "seeds 0 to 5"}[seeds]
    stage2 = ("then the 93-day validation span returned to the training set at the population's own sampling density (labels ending within "
              "24 h of the cut not returned) and training continued from the selected weights with a fresh AdamW for exactly 8 epochs, final "
              "weights kept")
    if recipe == "Rw":
        stage2 += ("; the loss of these 8 epochs weighted per training window by the frozen inverse-noise rule (omega = max(sigma^2, floor)^-1/2, "
                   "sigma^2 from the window's own context day tokens 338..365 against the same hours 7 days earlier, floor = the "
                   "draw-probability-weighted 25th percentile, weights normalised to a draw-weighted mean of 1 within each source population)")
    text = ("the copy-pointer carrier network (61,613 parameters) trained from scratch per population on a pooled training set (windows whose "
            "labels end before the population's validation set), early stopping on the population's last 93 days before the training cut "
            f"(at most 30 epochs, patience 5); {stage2}; {words} averaged at every issue time, anchored by the unchanged ANKYRA 2.2.1 "
            "anchoring layer, weekly seasonal naive on off-state and micro-load windows")
    if continuation:
        text += ("; then the origin-continuity layer (ankyra_lite.continuation, frozen constants) on the anchored trajectory of the other "
                 "windows, with energy, level and peak read before the layer (readout rule A)")
    return text


NOTE = ("Paths are relative to the research root (environment variable ANKYRA_LITE_RESEARCH_ROOT, default <repo>/../HKAD_V3). 'npz' is the "
        "anchored product before the continuation layer (the 0.1.0 format: building, origin, traj, level, peak, off, micro; plus state and "
        "the layer's pre-origin inputs cont_pre672, cont_last_type, cont_day_types); 'continued_npz' the same product with the layer "
        "(traj = the delivered trajectory, traj_anchored = the trajectory before the layer, level and peak unchanged, cont_* = the layer's "
        "record); 'group_npz' / 'group_continued_npz' the two three-seed groups of a six-seed configuration (seeds 0-2, 3-5), before and "
        "after the layer; 'run_dirs' the stage-2 run of every seed (carrier.npz: the carrier forecasts at the origin and the six "
        "pseudo-origins and the training cut; COMPLETE.json, TRAINED.json; the weights file named by 'checkpoint'); 'start_run_dirs' the "
        "from-scratch stage each run continued (best.pt); 'weights_record' the frozen loss weights of recipe Rw; 'span_plan' the returned "
        "validation span; 'prep' temp_sigma_std and the split record; 'recipe_variants' the other configurations and the 0.1.x recipe arms "
        "on the same windows. 'store': 'original' = the population's store as first built; 'corrected' = trained with the corrected CINELDI "
        "store (for CINELDI its own data, CINELDI origins = original origins + 1,416 h; for Drammen the CINELDI windows of its pooled "
        "training set).")


def variant_keys(recipe, seeds, continuation):
    """recipe_variants of a 0.2.0 configuration: every other study product (R_g0 = the 0.1.0 product), not the configuration
    itself and not its pre-layer product when the layer is on (the tables read that one from 'npz')."""
    own = product_name(recipe, seeds)
    keys = []
    for r in ("R", "Rw"):
        for n in (3, 6):
            nm = product_name(r, n)
            for c in (False, True):
                k = nm + ("_c" if c else "")
                if nm == own and (c == continuation or continuation):
                    continue
                keys.append(k)
    return keys


def build(recipe, seeds, continuation, log=print):
    """(mapping, problems).  The mapping is complete only when problems is empty."""
    from ankyra_lite import continuation as CT
    old = _json(MAPPING_0_1)
    name = product_name(recipe, seeds)
    seed_list = list(range(seeds))
    problems, checks = [], {"products": 0, "continued_products": 0, "carriers": 0, "checkpoints": 0, "start_runs": 0, "weights_records": 0}
    cst = {"kappa1": CT.CONSTANTS.kappa1, "rho1": CT.CONSTANTS.rho1, "kappa2": CT.CONSTANTS.kappa2, "rho2": CT.CONSTANTS.rho2}
    sha_cache = {}

    def sha(p):
        p = Path(p)
        if p not in sha_cache:
            sha_cache[p] = sha256(p)
        return sha_cache[p]

    def need(r):
        p = research(r)
        if not p.exists():
            problems.append(f"missing: {r}")
            return None
        return p

    def check_product(face, r, want_recipe, want_seeds, run_dirs, ref0):
        p = need(r)
        j = need(r[:-4] + ".json")
        if p is None or j is None:
            return None
        rec = _json(j)
        h = sha(p)
        if rec.get("sha256") != h:
            problems.append(f"{r}: SHA-256 differs from its record")
        if rec.get("recipe") != want_recipe or list(rec.get("carrier_seeds", [])) != list(want_seeds):
            problems.append(f"{r}: record names recipe {rec.get('recipe')} seeds {rec.get('carrier_seeds')}")
        cs = rec.get("carriers_sha256", {})
        for s in want_seeds:
            rd = run_dirs.get(s)
            if rd is None or str(s) not in cs:
                problems.append(f"{r}: no carrier record for seed {s}")
                continue
            c = research(rd) / "carrier.npz"
            if not c.exists() or sha(c) != cs[str(s)]:
                problems.append(f"{r}: the carrier forecasts of seed {s} are not those of {rd}")
            else:
                checks["carriers"] += 1
        z = np.load(p, allow_pickle=False)
        if not (np.array_equal(z["building"], ref0["building"]) and np.array_equal(z["origin"], ref0["origin"])
                and np.array_equal(np.asarray(z["off"]) | np.asarray(z["micro"]), ref0["state"])):
            problems.append(f"{r}: windows or state flags differ from the 0.1.0 product")
        checks["products"] += 1
        return h

    def check_continued(r, src, src_sha):
        p = need(r)
        j = need(r[:-4] + ".json")
        if p is None or j is None:
            return None
        rec = _json(j)
        h = sha(p)
        if rec.get("sha256") != h:
            problems.append(f"{r}: SHA-256 differs from its record")
        if rec.get("input_sha256") != src_sha:
            problems.append(f"{r}: not made from {src}")
        if rec.get("constants") != cst or rec.get("rule_digest") != CT.DIGEST:
            problems.append(f"{r}: made with other constants or another rule than ankyra_lite.continuation")
        checks["continued_products"] += 1
        return h

    faces = {}
    for face in NINE:
        e0 = old["faces"][face]
        z0 = np.load(research(e0["npz"]), allow_pickle=False)
        ref0 = {"building": z0["building"], "origin": z0["origin"], "state": np.asarray(z0["off"]) | np.asarray(z0["micro"])}
        del z0
        # stage-2 runs of every seed
        runs = {}
        for s in range(6):
            if recipe == "Rw":
                runs[s] = f"{STUDY}/runs/formal/{face}/Rw_s{s}"
            else:
                runs[s] = e0["run_dirs"][s] if s < 3 else f"{STUDY}/runs/formal/{face}/R_s{s}"
        runs_R = {s: (e0["run_dirs"][s] if s < 3 else f"{STUDY}/runs/formal/{face}/R_s{s}") for s in range(6)}
        runs_Rw = {s: f"{STUDY}/runs/formal/{face}/Rw_s{s}" for s in range(6)}
        starts = []
        for s in seed_list:
            rd = need(runs[s])
            if rd is None:
                starts.append(None)
                continue
            comp = _json(rd / "COMPLETE.json") if (rd / "COMPLETE.json").exists() else {}
            ck = rd / "final.pt"
            if not ck.exists() or sha(ck) != comp.get("checkpoint_sha256"):
                problems.append(f"{runs[s]}: final.pt missing or differs from COMPLETE.json")
            else:
                checks["checkpoints"] += 1
            st = _json(rd / "STARTED.json") if (rd / "STARTED.json").exists() else {}
            if s < 3 and recipe == "R":
                sd = e0["start_run_dirs"][s]
            else:
                t = _tail(st.get("start_checkpoint", ""))
                sd = str(Path(t).parent.as_posix()) if t else None
            if sd is None or not research(sd).exists():
                problems.append(f"{runs[s]}: from-scratch stage not found")
                starts.append(None)
                continue
            best = research(sd) / "best.pt"
            want = st.get("start_checkpoint_sha256")
            if not best.exists() or (want and sha(best) != want):
                problems.append(f"{runs[s]}: did not start from {sd}/best.pt")
            else:
                checks["start_runs"] += 1
            if recipe == "Rw":
                wr = f"{STUDY}/prep/formal/{face}/weights_Rw.json"
                wp = need(wr)
                if wp is not None and _json(wp).get("weights_sha256") != st.get("weights_sha256"):
                    problems.append(f"{runs[s]}: trained with other weights than {wr}")
            starts.append(sd)
        # the configuration's products
        npz = late(face, name)
        h = check_product(face, npz, recipe, seed_list, runs, ref0)
        e = {"store": e0["store"], "npz": npz, "npz_sha256": h}
        if continuation:
            e["continued_npz"] = late_cont(face, name)
            e["continued_npz_sha256"] = check_continued(e["continued_npz"], npz, h)
        if seeds == 6:
            e["group_npz"], e["group_continued_npz"] = [], []
            for g, members in enumerate(GROUPS6):
                gn = f"{recipe}_g{g}"
                gh = check_product(face, late(face, gn), recipe, members, runs, ref0)
                e["group_npz"].append(late(face, gn))
                if continuation:
                    check_continued(late_cont(face, gn), late(face, gn), gh)
                    e["group_continued_npz"].append(late_cont(face, gn))
            if not continuation:
                del e["group_continued_npz"]
        e["run_dirs"] = [runs[s] for s in seed_list]
        e["start_run_dirs"] = starts
        e["span_plan"], e["prep"] = e0["span_plan"], e0["prep"]
        if recipe == "Rw":
            wr = f"{STUDY}/prep/formal/{face}/weights_Rw.json"
            wp = research(wr)
            if wp.exists():
                w = _json(wp)
                e["weights_record"] = wr
                e["weights_record_sha256"] = sha(wp)
                e["weights"] = {k: w.get(k) for k in ("gamma", "floor_quantile", "floor_sigma2", "kish_ess_ratio_draws", "weight_max_over_median",
                                                      "weights_sha256")}
                checks["weights_records"] += 1
        # the other configurations on the same windows
        var = {}
        for k in variant_keys(recipe, seeds, continuation):
            base = k[:-2] if k.endswith("_c") else k
            r_, n_ = base.split("_")[0], 6 if base.endswith("ens6") else 3
            members = list(range(n_))
            rr = runs_R if r_ == "R" else runs_Rw
            if k == "R_g0":
                var[k] = e0["npz"]                                         # the 0.1.0 product itself
                fr = need(late(face, "R_g0")[:-4] + ".json")
                if fr is not None and not (_json(fr).get("reproduces_0_1_0") or {}).get("bitwise_equal"):
                    problems.append(f"{face}: R_g0 does not reproduce the 0.1.0 product bit for bit")
                continue
            if k.endswith("_c"):
                src = late(face, base)
                if need(src) is None:
                    continue
                check_continued(late_cont(face, base), src, sha(research(src)))
                var[k] = late_cont(face, base)
            else:
                check_product(face, late(face, base), r_, members, rr, ref0)
                var[k] = late(face, base)
        for k in OLD_ARMS:
            if k in e0.get("recipe_variants", {}):
                var[k] = e0["recipe_variants"][k]
        e["recipe_variants"] = var
        faces[face] = e
        log(f"{face}: {name}{' + layer' if continuation else ''}; {len(problems)} problem(s) so far")
    m = {"label": LABEL, "version": VERSION, "configuration": config_code(recipe, seeds, continuation),
         "configuration_parts": {"recipe": recipe, "seeds": seeds, "continuation": bool(continuation)},
         "recipe": RECIPE_NAME[recipe], "recipe_description": describe(recipe, seeds, continuation), "note": NOTE,
         "seeds": seed_list}
    if seeds == 6:
        m["seed_groups"] = [list(g) for g in GROUPS6]
    m.update({"checkpoint": "final.pt", "start_checkpoint": "best.pt",
              "continuation": {"enabled": bool(continuation), "readout_rule": "energy, level and peak before the layer (readout rule A)",
                               "constants": cst, "rule_digest": CT.DIGEST, "module": "ankyra_lite.continuation"},
              "checkpoint_folder": "rw" if recipe == "Rw" else "",
              "generated": {"tool": "tools/products_for_configuration.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks},
              "faces": faces, "replacements": {}})
    return m, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", required=True, choices=("R", "Rw", "r010"))
    ap.add_argument("--seeds", type=int, choices=(3, 6), default=6)
    ap.add_argument("--continuation", choices=("on", "off"), default="on")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output file (default: evaluation/products_v020_candidate.json)")
    ap.add_argument("--check", action="store_true", help="build and check only; nothing is written")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if CB._in_read_only(out):
        raise SystemExit("the output file lies in a read-only tree")
    if a.recipe == "r010":
        text = MAPPING_0_1.read_bytes()
        if not a.check:
            tmp = out.with_name(out.name + ".part")
            tmp.write_bytes(text)
            os.replace(tmp, out)
        print(f"0.1.x mapping {'checked' if a.check else 'written to ' + out.name} (sha256 {hashlib.sha256(text).hexdigest()[:12]})")
        return 0
    m, problems = build(a.recipe, a.seeds, a.continuation == "on")
    if problems:
        print(f"{len(problems)} problem(s); the mapping is not written:")
        for p in problems[:60]:
            print("  " + p)
        return 1
    if not a.check:
        text = json.dumps(m, indent=1, ensure_ascii=False) + "\n"
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(CB.scrub(text).encode("utf-8"))
        os.replace(tmp, out)
    print(f"{m['configuration']}: {len(m['faces'])} populations, checks {m['generated']['checks']}; "
          + ("checked, nothing written" if a.check else f"written to {out.name}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
