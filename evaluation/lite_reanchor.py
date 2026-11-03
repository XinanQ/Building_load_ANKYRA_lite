"""Re-anchoring of ankyra-lite's saved carrier forecasts with the unchanged anchoring layer (no training, no new carrier
forecast).  Used by the ablation, seed-spread, component and anchoring-record tables of compare_baselines.py.

For each population of the products mapping and each mode, the carrier forecasts at the origin and the six
pseudo-origins (the run directories' carrier.npz, the same arrays the evaluated product was anchored with) are given to
``ankyra_lite.anchor.forecast`` window by window, exactly as the study's anchoring step did
(tools_smallnet_sr_formal_v1._anchor_work: same history record, temp_sigma_std, daylight-saving region, peak readout):

  ens         the ensemble of the mapping's seeds with the default anchoring layer: the evaluated product itself,
              re-run.  Gate: it must reproduce the stored product bit for bit; it also records the anchoring record
              (``anchoring_record``)
  s<k>        one seed's carrier alone, default anchoring layer (k = each seed of the mapping).  Where the mapping lists
              the stored single-seed products (``seed_npz``) or the two three-seed groups (``group_npz``, six seeds),
              the tables read those and these modes are not run in full; a run on the first windows only (``--limit``,
              cache name ``s<k>_first<n>``) is then a gate: it must reproduce the stored single-seed product bit for bit
  g<j>        a six-seed mapping's three-seed group j (seeds 0-2, 3-5; ``seed_groups``), default anchoring layer.  Run
              on the first windows only as a gate (cache name ``g<j>_first<n>``): it must reproduce the stored group
              product, and with the continuation layer its continued product, bit for bit
  no_within   ensemble, ``within_anchor=False`` (the carrier's within-day block is kept)
  lead_trust  ensemble, ``single_trust=False`` (within-day trust per lead block instead of one value per window)
  no_gap      ensemble, ``gap_tolerance=False`` (no short-gap rule in the pseudo-origin bookkeeping)
  no_micro    ensemble, ``micro_load_rule=False`` (identical by construction where no window is a micro-load window;
              then not computed)
  fields      the ensemble call of ``ens`` again, keeping the anchoring layer's fields of every window
              (``daily_means_kw``, ``fixed_division_daily_means_kw``, ``within_day_kw``, ``foundation_within_day_kw``,
              ``analog_shape_kw``, ``lead_week_weights``, ``pseudo_pairs``, ``within_trust``, ``within_pseudo_pairs``,
              ``analog_kept``, ``level_weights``) and quantities rebuilt from the layer's public functions and the
              private helpers ``_masked`` and ``_level_parts`` (read only): the six handover pairs (fixed division,
              realised and carrier daily means at the completed pseudo-origins, ``history.estimate_from_history``), the
              three within-day triples (``analog.AnalogShapes``), the history-side daily means before the handover m^H,
              the seven level candidates at the origin, and the pseudo-origin sums of one whole-window combination weight
              for the history side and for three generic forecasters.  Gate: the delivered trajectory, level and state
              flags equal the stored product bit for bit, and the rebuilt pairs, triples and level reproduce the call's
              ``lead_week_weights``, ``within_trust``, ``level_weights`` and ``daily_means_kw`` exactly.  It is not
              an ablation; the tables of ``extra_tables.py`` read it (handover, lead weeks, within-day trust, whole-window
              weight, generic combinations, level-candidate hit rate)

Continuation layer (a mapping whose ``continuation`` is enabled, version 0.2.0).  Every mode also stores the layered
trajectory ``traj_cont``: ``ankyra_lite.continuation.apply_batch`` on the mode's float32 trajectory, with the layer's
inputs (the 672 load hours before the origin, the type of the last observed day, the types of forecast days 1-7) and the
mode's own state flags, returned as float32 like the study's continued products; its record (``cont_*``) is stored too.
The ``ens`` and ``fields`` gates then also require ``traj_cont`` to equal the mapping's continued product bit for bit and
the layer's inputs to equal those embedded in the product.

The seasonal-naive state rule is applied when the tables are scored, not here.  No target is read: the population's
record is read from the store files alone (store_arrays: the arrays of the study's loader tools_smallnet_carrier_v1.raw, without
the earlier study's window arrays, saved forecasts and results files that loader opens for some populations), every unit's load at
and after its last test origin is set to NaN as soon as it is loaded, and each window reads ``load[b, :o]`` only (plus the
day types of its forecast month, a calendar). The cache lives in ANKYRA_LITE_EVAL_CACHE (default: the system temporary
directory), under a versioned namespace bound to implementation, store/prep/product/carrier inputs, mode specification,
seed groups and continuation constants. Payload hashes are checked on read; legacy label-only caches are ignored and
never overwritten.

Usage:  python -B evaluation/lite_reanchor.py --faces oslo [--modes ens,s0] [--limit 8] [--workers 2] [--recipe p3s]
        python -B evaluation/lite_reanchor.py --faces oslo --products <mapping.json>
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import compare_baselines as CB                                     # noqa: E402  (environment, read-only guard, paths)

import argparse                                                    # noqa: E402
import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import math                                                        # noqa: E402
import re                                                          # noqa: E402
import time                                                        # noqa: E402
import warnings                                                    # noqa: E402
from concurrent.futures import ProcessPoolExecutor                 # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

MODES = {"ens": (None, {}), "s0": ((0,), {}), "s1": ((1,), {}), "s2": ((2,), {}),
         "no_within": (None, {"within_anchor": False}), "lead_trust": (None, {"single_trust": False}),
         "no_gap": (None, {"gap_tolerance": False}), "no_micro": (None, {"micro_load_rule": False})}      # a three-seed mapping
ABLATION_MODES = ("no_within", "lead_trust", "no_gap", "no_micro")
FIELD_MODES = {"fields": (None, {})}             # not an ablation: read by the handover and combination tables only
SPECS = MODES | FIELD_MODES                      # every mode of a three-seed mapping
LABELS = {"ens": "three-seed ensemble (the product, re-run)", "s0": "carrier of seed 0 alone", "s1": "carrier of seed 1 alone",
          "s2": "carrier of seed 2 alone", "no_within": "anchoring layer without the within-day anchoring",
          "lead_trust": "within-day trust per lead block instead of one value", "no_gap": "anchoring layer without the short-gap rule",
          "no_micro": "anchoring layer without the micro-load rule",
          "fields": "three-seed ensemble (the product, re-run) with the anchoring layer's fields and the rebuilt pseudo-origin quantities"}
STATS = ("carrier_level_weight", "history_vs_carrier_log_ratio", "pseudo_origins")
SEED_MODES = ("s0", "s1", "s2")
K, HORIZON, CONTEXT, HR = 6, CB.HORIZON, CB.CONTEXT, CB.HR
DAYS = CB.DAYS
CANDIDATES = ("s_u", "l_u", "a_u", "s_w", "l_w", "a_w", "fm")   # the seven level candidates (level_weights keys)
GENERIC = ("SN-week", "WeekProf4", "LYR-profile")             # generic partners of the whole-window combination
YEAR_LAG = 364 * 24                                            # the last-year profile's lag (the panel's definition)
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
CONT_INFO = ("xf", "xd", "s", "eps", "missing_f", "missing_d", "clipped_f", "clipped_d", "nonpositive_f", "nonpositive_d", "matched",
             "bypass_state", "d_star")                          # the layer's per-window record, stored as cont_<key>
# per-window arrays of the fields mode: name -> (shape after the window axis, dtype, fill)
FIELD_ARRAYS = {
    "daily": ((DAYS,), np.float64, np.nan),              # daily_means_kw (after the handover)
    "fixed": ((DAYS,), np.float64, np.nan),              # fixed_division_daily_means_kw
    "mh": ((DAYS,), np.float64, np.nan),                 # history-side daily means before the handover (level with the carrier candidate + daily path)
    "within": ((HORIZON,), np.float64, np.nan),          # within_day_kw (delivered within-day block)
    "fwithin": ((HORIZON,), np.float64, np.nan),         # foundation_within_day_kw (the carrier's within-day block)
    "analog": ((HORIZON,), np.float64, np.nan),          # analog_shape_kw (S0; the carrier's block when the guard fell back)
    "alphas": ((4,), np.float64, np.nan),                # lead_week_weights
    "alpha_month": ((), np.float64, np.nan),             # one weight per window from the same pairs (the 31 days pooled)
    "npairs": ((), np.int64, 0), "trust": ((4,), np.float64, np.nan), "ntrip": ((), np.int64, 0),
    "kept": ((), bool, False), "lw": ((len(CANDIDATES),), np.float64, np.nan),
    "cand": ((len(CANDIDATES),), np.float64, np.nan),    # each level candidate's window mean at the origin (kW)
    "b2_sums": ((4,), np.float64, np.nan),               # daily num, den (handover pairs), within-day num, den (trust triples)
    "gen_sums": ((len(GENERIC), 4), np.float64, np.nan),  # the same sums with a generic forecaster in place of the history side
    "fields_ok": ((), bool, False),                      # a non-state window whose fields were recorded
    "rebuild_ok": ((), bool, False),                     # the rebuilt quantities reproduce the call's fields exactly
}
CACHE_SCHEMA = "ankyra-lite-reanchor-cache-v2"
CACHE_IMPLEMENTATION_REV = "2026-10-22-cache-identity-1"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_key(prods):
    """The cache folder's key of a products mapping: its label, and its configuration when it names one (0.2.0), so that
    the configurations of one version keep separate caches."""
    return prods["label"] + (f" {prods['configuration']}" if prods.get("configuration") else "")


def cache_identity(*, label, face, mode, limit, spec, layer, seeds, seed_groups, constants_digest,
                   input_manifest, implementation_digest):
    """Canonical content identity for a re-anchoring cache entry; pure and safe to test with synthetic manifests."""
    payload = {"schema": CACHE_SCHEMA, "implementation_revision": CACHE_IMPLEMENTATION_REV,
               "implementation_digest": str(implementation_digest), "label": str(label), "face": str(face),
               "mode": str(mode), "limit": None if limit is None else int(limit),
               "spec": {"seeds": None if spec[0] is None else [int(x) for x in spec[0]],
                        "kwargs": dict(spec[1])}, "layer": bool(layer), "constants_digest": constants_digest,
               "seeds": [int(x) for x in seeds], "seed_groups": [[int(x) for x in g] for g in seed_groups],
               "inputs": sorted((dict(x) for x in input_manifest), key=lambda x: x["path"])}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"payload": payload, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def cache_paths(label, face, mode, limit, identity, cache_root=None):
    """Versioned namespace and paths for one identity; legacy path layout is never consulted."""
    name = mode if limit is None else f"{mode}_first{int(limit)}"
    root = Path(cache_root or CB.CACHE) / CACHE_SCHEMA / identity["sha256"] / re.sub(r"[^A-Za-z0-9._-]+", "_", label) / face
    data = root / f"{name}.npz"
    return data, data.with_suffix(".json")


def cache_file(label, face, mode, limit=None, identity=None):
    """Compatibility helper; an explicit full identity is required for all cache access."""
    if identity is None:
        raise ValueError("cache_file requires a complete cache identity")
    return cache_paths(label, face, mode, limit, identity)[0]


def _source_store_paths(face, entry):
    """Resolve only source file paths from study metadata; file bytes are fingerprinted without loading arrays."""
    _paths()
    import tools_hcr_within_anchor_r3_v1 as R3
    P = R3.P
    if face == "cineldi" and entry.get("store") == "corrected":
        store, _ = CB.corrected_store_info()
        original_dir, _ = R3.PA.SETS[face]
        return [Path(original_dir) / "store.npz", Path(store) / "store.npz"]
    if face in R3.PA.SETS:
        store_dir, _ = R3.PA.SETS[face]
        return [Path(store_dir) / "store.npz"]
    if face in P.EXT:
        return [Path(P.EXT[face][0])]
    if face in P.EARLY:
        return [Path(x) for x in P.EARLY[face][0]]
    if face == "goiener_confirm":
        return [Path(P.GOI)]
    raise ValueError(f"no auditable source-store path for {face!r}")


def _cache_file_manifest(face, entry, mode, spec, layer):
    files = {"store": _source_store_paths(face, entry), "prep": [Path(entry["prep_path"])],
             "product": [Path(entry["npz_path"])], "carrier_runs": [Path(rd) / "carrier.npz" for rd in
                 (entry["run_paths"] if spec[0] is None else
                  [entry["run_paths"][seeds_of(entry).index(int(s))] for s in spec[0]])]}
    if layer and entry.get("continued_path") is not None:
        files["continued_product"] = [Path(entry["continued_path"])]
    if face == "cineldi" and entry.get("store") == "corrected":
        files["store_selector"] = [Path(CB.DC_BUILD)]
    if is_seed_mode(mode) and entry.get("seed_paths"):
        k = seeds_of(entry).index(int(mode[1:]))
        files["seed_gate"] = [Path(entry["seed_paths"][k])]
    if is_group_mode(mode) and entry.get("group_paths"):
        j = int(mode[1:])
        files["group_gate"] = [Path(entry["group_paths"][j])]
        if layer and entry.get("group_continued_paths"):
            files["group_continued_gate"] = [Path(entry["group_continued_paths"][j])]
    result = []
    for role, paths in sorted(files.items()):
        for path in paths:
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"cache input {role} is missing: {path}")
            result.append({"role": role, "path": path.as_posix(), "size": path.stat().st_size, "sha256": _sha(path)})
    return result


def _implementation_digest(layer):
    _paths()
    import tools_smallnet_carrier_v1 as V
    import tools_hcr_within_anchor_r3_v1 as R3
    import tools_hcr_analog_v2 as A2
    import tools_hcr_ext_drammen_score_v1 as DS
    import tools_hcr_level_zero_v1 as LZ
    import tools_release_repro_check_v5 as RC
    paths = [Path(__file__), Path(CB.__file__), Path(V.__file__), Path(R3.__file__), Path(R3.P.__file__),
             Path(R3.PA.__file__), Path(A2.__file__), Path(DS.__file__), Path(LZ.__file__), Path(RC.__file__)]
    anchor = Path(__file__).resolve().parents[1] / "ankyra_lite" / "anchor"
    paths.extend(sorted((*anchor.rglob("*.py"), *anchor.rglob("*.json"))))
    if layer:
        paths.append(Path(__file__).resolve().parents[1] / "ankyra_lite" / "continuation.py")
    h = hashlib.sha256()
    for path in paths:
        path = path.resolve()
        h.update(path.as_posix().encode("utf-8")); h.update(_sha(path).encode("ascii"))
    return h.hexdigest()


def _identity_for(label, face, mode, entry, limit):
    spec = modes_for(entry)[mode]
    layer = layer_on(entry)
    constants_digest = None
    if layer:
        from ankyra_lite import continuation as CT
        constants_digest = CT.DIGEST
    return cache_identity(label=label, face=face, mode=mode, limit=limit, spec=spec, layer=layer,
                          seeds=seeds_of(entry), seed_groups=groups_of(entry), constants_digest=constants_digest,
                          input_manifest=_cache_file_manifest(face, entry, mode, spec, layer),
                          implementation_digest=_implementation_digest(layer))


def validate_cache(data_path, meta_path, identity, *, allow_identical=False):
    """Fail closed on stale, partial, mismatched, or damaged cache entries."""
    data_path, meta_path = Path(data_path), Path(meta_path)
    if not meta_path.exists():
        if data_path.exists():
            raise RuntimeError(f"unbound cache payload exists: {data_path}")
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid cache metadata: {meta_path}") from exc
    if meta.get("cache_identity") != identity["payload"] or meta.get("cache_identity_sha256") != identity["sha256"]:
        raise RuntimeError(f"cache identity mismatch: {meta_path}")
    if meta.get("identical_to"):
        if (identity["payload"].get("mode") != "no_micro" or meta.get("identical_to") != "ens"
                or data_path.exists() or not allow_identical):
            raise RuntimeError(f"invalid identical-to cache record: {meta_path}")
        return meta
    if not data_path.exists() or meta.get("payload_sha256") != _sha(data_path):
        raise RuntimeError(f"cache payload is missing or damaged: {data_path}")
    return meta


def claim_cache(data_path, identity):
    """Claim a cache miss exclusively; leave the claim behind if computation fails."""
    data_path = Path(data_path)
    claim_path = data_path.with_suffix(data_path.suffix + ".claim")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"cache_identity": identity["payload"], "cache_identity_sha256": identity["sha256"],
              "status": "claimed"}
    try:
        with claim_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(f"cache identity is already claimed; refusing concurrent or interrupted recompute: {claim_path}") from exc
    return claim_path


def complete_cache_claim(claim_path):
    """Mark a successfully written cache claim complete without removing its audit record."""
    claim_path = Path(claim_path)
    record = json.loads(claim_path.read_text(encoding="utf-8"))
    if record.get("status") != "claimed":
        raise RuntimeError(f"invalid cache claim state: {claim_path}")
    record["status"] = "completed"
    tmp = claim_path.with_name(claim_path.name + ".part")
    tmp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, claim_path)


def cached_gate_records(cache_root, label, face, prefix, identity_factory):
    """Find and validate first-N seed/group gates across fingerprint namespaces; never scans legacy caches."""
    root = Path(cache_root) / CACHE_SCHEMA
    clean_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    records = {}
    if not root.exists():
        return records
    for meta_path in sorted(root.glob(f"*/{clean_label}/{face}/{prefix}*_first*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mode = str(meta.get("mode", ""))
        limit = meta.get("limit")
        if not mode.startswith(prefix) or limit is None:
            continue
        identity = identity_factory(mode, int(limit))
        data_path, expected_meta = cache_paths(label, face, mode, int(limit), identity, cache_root=cache_root)
        if meta_path.resolve() != expected_meta.resolve():
            continue
        checked = validate_cache(data_path, meta_path, identity, allow_identical=True)
        if checked and checked.get("gate"):
            records[meta_path.stem] = checked["gate"]
    return records


def seeds_of(entry):
    return [int(s) for s in entry.get("seeds", (0, 1, 2))]


def groups_of(entry):
    """The three-seed groups of a six-seed mapping (one list of seeds per stored group product), else []."""
    return [list(g) for g in entry.get("seed_groups") or []] if entry.get("group_paths") else []


def layer_on(entry):
    return bool(entry.get("layer"))


def modes_for(entry):
    """{mode: (seeds or None, anchoring kwargs)} of every mode this mapping entry accepts."""
    out = {"ens": (None, {})}
    for s in seeds_of(entry):
        out[f"s{s}"] = ((s,), {})
    for j, g in enumerate(groups_of(entry)):
        out[f"g{j}"] = (tuple(g), {})
    for m in ABLATION_MODES:
        out[m] = MODES[m]
    return out | FIELD_MODES


def is_seed_mode(mode):
    return bool(re.fullmatch(r"s\d+", mode))


def is_group_mode(mode):
    return bool(re.fullmatch(r"g\d+", mode))


def label_of(mode, entry):
    n = len(seeds_of(entry))
    words = COUNT_WORDS.get(n, str(n))
    if mode == "ens":
        return f"{words}-seed ensemble (the product, re-run)"
    if mode == "fields":
        return f"{words}-seed ensemble (the product, re-run) with the anchoring layer's fields and the rebuilt pseudo-origin quantities"
    if is_seed_mode(mode):
        return f"carrier of seed {mode[1:]} alone"
    if is_group_mode(mode):
        g = groups_of(entry)[int(mode[1:])]
        return f"three-seed group {mode[1:]} (seeds {g[0]}-{g[-1]})"
    return LABELS[mode]


def default_modes(entry):
    """All modes, except the single seeds when the mapping lists stored single-seed products or the stored three-seed
    groups (they are read instead), and the groups themselves (gates only)."""
    stored = bool(entry.get("seed_paths") or entry.get("group_paths"))
    return [m for m in modes_for(entry) if m not in FIELD_MODES and not is_group_mode(m) and not (is_seed_mode(m) and stored)]


def _gate(traj, level, peak, off, micro, zx, n):
    g = {"traj_equal": bool(np.array_equal(traj, zx["traj"][:n])),
         "traj_max_abs_diff": float(np.nanmax(np.abs(traj.astype(np.float64) - zx["traj"][:n]))),
         "level_equal": bool(np.array_equal(level, zx["level"][:n])), "peak_equal": bool(np.array_equal(peak, zx["peak"][:n])),
         "state_flags_equal": bool(np.array_equal(off, zx["off"][:n]) and np.array_equal(micro, zx["micro"][:n])), "windows": int(n)}
    g["pass"] = bool(g["traj_equal"] and g["level_equal"] and g["state_flags_equal"])
    return g


def _gate_layer(traj_cont, zc, n, inputs_equal):
    """The layered trajectory against a continued product (float32, bit for bit) and the layer's inputs against those the
    product embeds."""
    g = {"continued_traj_equal": bool(np.array_equal(traj_cont, zc["traj"][:n])),
         "continued_traj_max_abs_diff": float(np.nanmax(np.abs(traj_cont.astype(np.float64) - zc["traj"][:n]))),
         "layer_inputs_equal_embedded": inputs_equal}
    g["layer_pass"] = bool(g["continued_traj_equal"] and inputs_equal is not False)
    return g


def _paths():
    for p in (str(CB.REPO), str(CB.REL), str(CB.HKAD)):
        if p not in sys.path:
            sys.path.insert(0, p)


def store_arrays(face):
    """The store arrays of one population (load, temp, types, names, groups, start_iso) exactly as the study's loader
    tools_smallnet_carrier_v1.raw returns them, but opening the store files only.  For Cambridge, HEEW and CINELDI the
    study's loader opens the store alone and is called as it is; for the other populations it goes through the panel's
    loader (tools_hcr_f1_panel_v1.load_face), which also opens the earlier study's window arrays (with their targets
    and losses), its saved forecasts and its results file, and for BDG2 selects windows from the observed hours
    after each origin.  Those reads are left out here; the arrays kept are built by the same steps from the same store
    files (tests/test_store_arrays.py: equal to the study's loader on the nine populations)."""
    _paths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tools_smallnet_carrier_v1 as V
        import tools_hcr_within_anchor_r3_v1 as R3
    P = R3.P
    if face in R3.PA.SETS:                                       # the study's loader reads the store alone
        return V.raw(face)
    if face in P.EXT:
        sp, _, rule = P.EXT[face]
        z = np.load(sp, allow_pickle=False)
        types = z["types"].astype(np.int64)
        if types.ndim == 2:
            if not (types == types[:1]).all():
                raise SystemExit(f"{face}: day types differ between units")
            types = types[0]
        names = [str(x) for x in z["buildings"]]
        groups = ([P.GROUP_OF[str(c)] for c in z["categories"]] if rule == "cat" else [str(g) for g in z["groups"]] if rule == "groups"
                  else [rule] * len(names))
        start_iso = str(z["start_iso"]) if "start_iso" in z.files else P.START_ISO
        d = {"load": z["load"].astype(np.float64), "temp": z["temp"].astype(np.float32), "types": types, "names": names, "groups": groups,
             "start_iso": start_iso}
    elif face in P.EARLY:
        paths, start_iso, pad = P.EARLY[face]
        zs = [np.load(p, allow_pickle=False) for p in paths]
        load = np.concatenate([z["load"].astype(np.float64) for z in zs])
        temp = np.concatenate([z["temp"].astype(np.float32) for z in zs])
        types = np.concatenate([z["types"].astype(np.int64) for z in zs])
        names = [str(x) for z in zs for x in z["buildings"]]
        groups = [P.GROUPS[int(g)] for z in zs for g in z["groups"]]
        if len(set(names)) != len(names):
            raise SystemExit(f"{face}: duplicate unit names")
        if pad:
            t0 = P.iso_t0(start_iso)
            wd = np.array([(t0 + P.timedelta(hours=h)).weekday() for h in range(0, pad, 24)]).repeat(24)
            tpad = temp[:, 8760 - pad:8760]
            load = np.concatenate([np.full((len(load), pad), np.nan), load], 1)
            temp = np.concatenate([tpad, temp], 1)
            types = np.concatenate([np.tile(wd, (len(types), 1)), types], 1)
        d = {"load": load, "temp": temp, "types": types, "names": names, "groups": groups, "start_iso": start_iso}
    else:                                                        # the GoiEner non-household store
        z = np.load(P.GOI, allow_pickle=False)
        d = {"load": z["load"].astype(np.float64), "temp": z["temp"].astype(np.float32), "types": z["types"].astype(np.int64),
             "names": [str(x) for x in z["buildings"]], "groups": [P.GROUPS[int(g)] for g in z["groups"]], "start_iso": P.START_ISO}
    d["load"] = np.asarray(d["load"], np.float64)                # the study's loader's last steps (tools_smallnet_carrier_v1.raw)
    d["temp"] = np.asarray(d["temp"], np.float32)
    d["types"] = np.asarray(d["types"], np.int64)
    d["groups"] = list(d["groups"])
    d["names"] = list(d["names"])
    return d


def history_only(load, wb, wo):
    """Set every unit's load at and after its last test origin to NaN (in place) and the rows of units without a test
    window to NaN: no hour that is a target of the last window, and no unit outside the test set, stays in memory."""
    last = {}
    for b, o in zip(np.asarray(wb, np.int64), np.asarray(wo, np.int64)):
        last[int(b)] = max(last.get(int(b), 0), int(o))
    for b in range(load.shape[0]):
        if b in last:
            load[b, last[b]:] = np.nan
        else:
            load[b, :] = np.nan
    return load


def face_data(face, entry, wb=None, wo=None):
    """The anchoring inputs of one population (load, temperature, day types, names, groups, start timestamp, region), as
    the study's anchoring step loaded them (tools_ankyra_simplify_v1.face), from the store files alone (store_arrays: the
    arrays of the study's store loader tools_smallnet_carrier_v1.raw); on the corrected CINELDI store,
    load, temperature, day types and start timestamp come from that store.  With the windows (wb, wo), the load at and
    after each unit's last test origin is set to NaN (history_only)."""
    _paths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tools_smallnet_carrier_v1 as V
    d = store_arrays(face)
    d["region"] = V._region(face, d)
    if face == "cineldi" and entry["store"] == "corrected":
        store, _ = CB.corrected_store_info()
        z = np.load(Path(store) / "store.npz", allow_pickle=False)
        d = dict(d)
        d["load"] = z["load"].astype(np.float64)
        d["temp"] = z["temp"].astype(np.float32)
        d["types"] = z["types"].astype(np.int64)
        d["start_iso"] = str(z["start_iso"])
    if wb is not None:
        d["load"] = history_only(d["load"], wb, wo)
    sigma = float(json.loads(Path(entry["prep_path"]).read_text(encoding="utf-8"))["sigma"])
    return d, sigma


def carriers(entry, seeds, wb, wo):
    """Sum in seed order in float64 over the chosen seeds, divided by their number (the study's ensemble), and the
    pseudo-origins every chosen seed has."""
    Sp, Sh = None, None
    order = seeds_of(entry)
    runs = entry["run_paths"] if seeds is None else [entry["run_paths"][order.index(int(s))] for s in seeds]
    for rd in runs:
        q = np.load(Path(rd) / "carrier.npz", allow_pickle=False)
        if not (np.array_equal(q["building"], wb) and np.array_equal(q["origin"], wo)):
            key = {(int(b), int(o)): i for i, (b, o) in enumerate(zip(q["building"], q["origin"]))}
            ix = np.array([key[(int(b), int(o))] for b, o in zip(wb, wo)])
            pred, have = q["pred"][ix], q["have"][ix]
        else:
            pred, have = q["pred"], q["have"]
        Sp = pred.astype(np.float64) if Sp is None else Sp + pred
        Sh = have if Sh is None else Sh & have
    return Sp / len(runs), Sh


_D = {}


def _work(args):
    face, mode, idx, entry, wb, wo, spec, layer = args
    _paths()
    import torch
    from ankyra_lite import anchor as A
    from ankyra_lite.anchor import readouts as RO
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    CT = None
    if layer:
        from ankyra_lite import continuation as CT
    if (face, mode) not in _D:
        _D.clear()
        d, sg = face_data(face, entry, wb, wo)
        Sp, Sh = carriers(entry, spec[0], wb, wo)
        _D[(face, mode)] = (d, sg, Sp, Sh)
    d, sg, Sp, Sh = _D[(face, mode)]
    kw = spec[1]
    out = []
    for i in idx:
        b, o = int(wb[i]), int(wo[i])
        pre_load = d["load"][b, :o]                                                # the only load this window reads
        ty = d["types"] if d["types"].ndim == 1 else d["types"][b]
        tmp = d["temp"][b] if d["temp"].ndim == 2 else d["temp"]
        h = A.History(load_kw=pre_load, temperature_c=tmp[:o], day_types=ty[:o + HORIZON], start_timestamp=d["start_iso"],
                      observed=np.isfinite(pre_load))
        F = {k: Sp[i, k] for k in range(K + 1) if Sh[i, k]}
        row = {"i": i, "err": ""}
        try:
            r = A.forecast(h, foundation=F, group=d["groups"][b], temp_sigma_std=sg, dst_region=d["region"](b), **kw)
        except Exception as e:                                                     # noqa: BLE001
            row["err"] = type(e).__name__
            out.append(row)
            continue
        ctx = pre_load[o - CONTEXT:o]
        ct = ty[o - CONTEXT:o:HR].astype(np.int64)
        tt = ty[o:o + HORIZON:HR].astype(np.int64)
        tr = np.asarray(r.trajectory_kw, np.float64)
        try:
            pk = float(RO.peak_readout(tr, ctx[None], ct[None], tt[None])[0])
        except Exception as e:                                                     # noqa: BLE001
            pk = float("nan")
            row["err"] += f"peak:{type(e).__name__};"
        rec = r.anchoring_record or {}
        row.update(traj=tr.astype(np.float32), level=float(r.level_kw), peak=pk, off=bool(r.off_state), micro=bool(r.micro_load),
                   stats=[float(rec.get(k, float("nan"))) for k in STATS])
        if CT is not None:                                                         # the continuation layer on the float32 trajectory
            pre, lt, dt = CT.window_inputs(pre_load, ty, o)
            try:
                Fp, info = CT.apply_batch(row["traj"][None], pre[None], [lt], [dt], [row["off"] or row["micro"]])
                row.update(traj_cont=Fp[0].astype(np.float32), cont={k: info[k][0] for k in CONT_INFO}, cont_in=(pre, lt, tuple(dt)))
            except Exception as e:                                                 # noqa: BLE001
                row["err"] += f"layer:{type(e).__name__};"
        if mode in FIELD_MODES:
            try:
                row["fields"] = window_fields(h, F, r, d["groups"][b], sg, d["region"](b))
            except Exception as e:                                                 # noqa: BLE001
                row["ferr"] = f"{type(e).__name__}: {e}"
        out.append(row)
    return out


def x_forecast(name, load, q):
    """A generic forecaster issued at hour q from load[:q] only (the panel's definitions: seasonal naive of the last
    week, the mean of the last four weeks, the 744 hours one year (364 days) earlier, else the seasonal naive)."""
    ctx = load[q - CONTEXT:q]
    if name == "SN-week":
        return np.tile(ctx[-168:], math.ceil(HORIZON / 168))[:HORIZON].astype(np.float64)
    if name == "WeekProf4":
        return np.tile(ctx[-4 * 168:].reshape(4, 168).mean(0), math.ceil(HORIZON / 168))[:HORIZON].astype(np.float64)
    if name == "LYR-profile":
        s = q - YEAR_LAG
        if s >= 0:
            seg = load[s:s + HORIZON]
            if np.isfinite(seg).all():
                return seg.astype(np.float64)
        return x_forecast("SN-week", load, q)
    raise KeyError(name)


def window_fields(h, F, r, group, sigma, region):
    """The fields of one default anchoring call ``r`` (``ankyra_lite.anchor.forecast(h, foundation=F, ...)``) and the
    quantities rebuilt from the layer's own functions, exactly as ``core.forecast`` computes them: the masked record with
    short gaps filled, ``_level_parts`` (level candidates and the completed pseudo-origins), the six handover pairs
    (``history.estimate_from_history`` at each completed pseudo-origin) and the three within-day triples
    (``analog.AnalogShapes``).  ``rebuild_ok``: the rebuilt pairs give ``lead_week_weights``, the rebuilt triples
    ``within_trust``, the rebuilt level ``level_weights`` and the handover of m^H ``daily_means_kw``, all exactly.
    Reads no hour at or after the origin except the pseudo-origins' completed targets (all before the origin)."""
    import warnings as _w
    from dataclasses import replace
    from ankyra_lite.anchor import core as C, analog as AN, blocks as BL
    from ankyra_lite.anchor.history import History, estimate_from_history
    out = {k: (np.full(s, f, dt) if s else (dt(f) if dt is not bool else bool(f))) for k, (s, dt, f) in FIELD_ARRAYS.items()}
    if r.off_state or r.micro_load:
        return out
    hm = C._masked(h)
    load = C.fill_short_gaps(np.asarray(hm.load_kw, dtype=np.float64))
    obs = np.isfinite(load)
    hf = replace(hm, load_kw=load, observed=obs)
    temp = np.asarray(hf.temperature_c, dtype=np.float32)
    types = np.asarray(hf.day_types)
    o = len(load)
    T0 = np.asarray(F[0], dtype=np.float64)
    mT = T0.reshape(DAYS, HR).mean(1)
    parts = C._level_parts(hf, group, sigma, min_target_obs=C.TARGET_MIN_OBS)
    fm_ps = {k: float(F[k].mean()) for k in range(1, K + 1) if k in parts["pseudo"] and k in F}
    lvl, wl = C.level_with_model_candidate(parts, float(T0.mean()), fm_ps)
    fixed = np.asarray(r.fixed_division_daily_means_kw, dtype=np.float64)
    mh = fixed - fixed.mean() + lvl
    hh = parts["h"]
    cand = []
    for n in CANDIDATES[:-1]:
        v = float(hh[n][0]) if n in parts["names"] else float("nan")
        cand.append(float(parts["l0"] + parts["s0"] * (v + float(hh["c_bar"][0]) + (float(hh["g_h"][0].mean()) if n in ("s_w", "l_w", "a_w") else 0.0))))
    cand.append(float(T0.mean()))
    pairs, ks = [], []
    for k in range(1, K + 1):
        p = parts["pseudo"].get(k)
        if p is None or k not in F:
            continue
        ok = p["o"]
        ek = estimate_from_history(History(load_kw=load[:ok], temperature_c=temp[:ok], day_types=types[:ok + HORIZON],
                                           start_timestamp=hf.start_timestamp, observed=obs[:ok]), group=group, temp_sigma_std=sigma)
        yk = load[ok:ok + HORIZON].reshape(DAYS, HR)
        nk = np.isfinite(yk).sum(1)
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            yday = np.where(nk >= 12, np.nanmean(yk, axis=1), np.nan)
        pairs.append((yday, ek.level_kw + ek.daily_path_kw, np.asarray(F[k], dtype=np.float64).reshape(DAYS, HR).mean(1)))
        ks.append(k)
    aw = C.lead_week_weights(pairs) if pairs else [0.5] * 4
    num = den = 0.0
    for y_, hd, fd in pairs:                                           # the 31 days pooled (one weight per window)
        m = np.isfinite(y_)
        num += float(((y_ - hd)[m] * (fd - hd)[m]).sum())
        den += float(((fd - hd)[m] ** 2).sum())
    lam = min(max(num / den, 0.0), 1.0) if den > 0 else 0.5
    am = (len(pairs) * lam + C.K0_WEEK * 0.5) / (len(pairs) + C.K0_WEEK)
    wT0 = BL.within_day(T0)
    shapes = AN.AnalogShapes(load, temp, types, hf.start_timestamp, region, origin=o)
    S0, kept = shapes.shape_with_sanity(o, wT0)
    triples, tks = [], []
    for k in range(1, AN.K_WITHIN + 1):
        q = o - k * HORIZON
        if k not in F or q - CONTEXT < 0 or not (obs[q - CONTEXT:q + HORIZON].all() and np.isfinite(load[q - CONTEXT:q + HORIZON]).all()):
            continue
        wTq = BL.within_day(np.asarray(F[k], dtype=np.float64))
        Sq, _ = shapes.shape_with_sanity(q, wTq)
        triples.append((Sq, wTq, BL.within_day(load[q:q + HORIZON])))
        tks.append(k)
    wtr, n_w = AN.within_trust(triples, single=True)
    b2 = [num, den, sum(float((np.asarray(s_) - f_) @ (y_ - f_)) for s_, f_, y_ in triples),
          sum(float((np.asarray(s_) - f_) @ (np.asarray(s_) - f_)) for s_, f_, _ in triples)]
    gen = np.zeros((len(GENERIC), 4))
    for j, name in enumerate(GENERIC):
        for (y_, _, fd), k in zip(pairs, ks):
            xd = x_forecast(name, load, o - k * HORIZON).reshape(DAYS, HR).mean(1)
            m = np.isfinite(y_)
            gen[j, 0] += float(((y_ - xd)[m] * (fd - xd)[m]).sum())
            gen[j, 1] += float(((fd - xd)[m] ** 2).sum())
        for (_, f_, y_), k in zip(triples, tks):
            xw = BL.within_day(x_forecast(name, load, o - k * HORIZON))
            gen[j, 2] += float((y_ - xw) @ (f_ - xw))
            gen[j, 3] += float((f_ - xw) @ (f_ - xw))
    rebuild = (tuple(float(a) for a in aw) == tuple(r.lead_week_weights) and tuple(float(x) for x in wtr) == tuple(r.within_trust)
               and n_w == r.within_pseudo_pairs and len(pairs) == r.pseudo_pairs and dict(wl) == dict(r.level_weights)
               and np.array_equal(C.lead_week_transition(mh, mT, aw), np.asarray(r.daily_means_kw))
               and np.array_equal(np.asarray(S0), np.asarray(r.analog_shape_kw)) and bool(kept) == bool(r.analog_kept))
    out.update(daily=np.asarray(r.daily_means_kw, dtype=np.float64), fixed=fixed, mh=mh, within=np.asarray(r.within_day_kw, dtype=np.float64),
               fwithin=np.asarray(r.foundation_within_day_kw, dtype=np.float64),
               analog=np.asarray(r.analog_shape_kw, dtype=np.float64) if r.analog_shape_kw is not None else np.full(HORIZON, np.nan),
               alphas=np.asarray(r.lead_week_weights, dtype=np.float64), alpha_month=float(am), npairs=int(r.pseudo_pairs),
               trust=np.asarray(r.within_trust, dtype=np.float64), ntrip=int(r.within_pseudo_pairs), kept=bool(r.analog_kept),
               lw=np.array([float(r.level_weights.get(n, np.nan)) for n in CANDIDATES]), cand=np.array(cand), b2_sums=np.array(b2), gen_sums=gen,
               fields_ok=True, rebuild_ok=bool(rebuild))
    return out


def _inputs_equal(rows, zx, n):
    """The layer's inputs of the re-run against those the product embeds (cont_pre672, cont_last_type, cont_day_types);
    None when the product embeds none."""
    if "cont_pre672" not in zx.files:
        return None
    for r in rows:
        if "cont_in" not in r:
            continue
        i = r["i"]
        pre, lt, dt = r["cont_in"]
        if not (np.array_equal(pre, zx["cont_pre672"][i], equal_nan=True) and lt == int(zx["cont_last_type"][i])
                and tuple(int(x) for x in zx["cont_day_types"][i]) == dt):
            return False
    return True


def run_mode(face, mode, entry, label, workers=2, limit=None):
    """Anchor one population in one mode (cached).  Returns the record."""
    identity = _identity_for(label, face, mode, entry, limit)
    f, meta = cache_paths(label, face, mode, limit, identity)
    psha = _sha(entry["npz_path"])
    layer = layer_on(entry)
    cached = validate_cache(f, meta, identity, allow_identical=True)
    if cached is not None:
        if cached.get("product_sha256") != psha or cached.get("limit") != limit:
            raise RuntimeError(f"cache record fields disagree with identity: {meta}")
        return cached
    claim_path = claim_cache(f, identity)
    zx = np.load(entry["npz_path"], allow_pickle=False)
    wb, wo = zx["building"].astype(np.int64), zx["origin"].astype(np.int64)
    specs = modes_for(entry)
    spec = specs[mode]
    if mode == "no_micro" and not zx["micro"].any():
        m = {"face": face, "mode": mode, "label": label_of(mode, entry), "identical_to": "ens", "reason": "no micro-load window in the product",
             "product_sha256": psha, "limit": limit, "cache_identity": identity["payload"],
             "cache_identity_sha256": identity["sha256"]}
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp_meta = meta.with_name(meta.name + ".part")
        tmp_meta.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_meta, meta)
        complete_cache_claim(claim_path)
        return m
    n = len(wb) if limit is None else min(int(limit), len(wb))
    e = {k: v for k, v in entry.items() if k in ("store", "prep_path", "run_paths", "npz_path", "seeds")}
    e = {k: (str(v) if isinstance(v, Path) else ([str(x) for x in v] if k == "run_paths" else v)) for k, v in e.items()}
    t0 = time.time()
    res = [None] * n
    if workers <= 1:
        parts = [_work((face, mode, list(range(n)), e, wb, wo, spec, layer))]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_work, [(face, mode, list(range(s, n, workers)), e, wb, wo, spec, layer) for s in range(workers)]))
    for part in parts:
        for r in part:
            res[r["i"]] = r
    errs = [(r["i"], r["err"]) for r in res if r["err"]]
    ok = [r for r in res if not r["err"]]
    traj = np.full((n, HORIZON), np.nan, np.float32)
    level, peak = np.full(n, np.nan), np.full(n, np.nan)
    off, micro = np.zeros(n, bool), np.zeros(n, bool)
    stats = np.full((n, len(STATS)), np.nan)
    for r in ok:
        i = r["i"]
        traj[i], level[i], peak[i], off[i], micro[i], stats[i] = r["traj"], r["level"], r["peak"], r["off"], r["micro"], r["stats"]
    extra, ferrs = {}, []
    if layer:
        extra["traj_cont"] = np.full((n, HORIZON), np.nan, np.float32)
        proto = {"xf": np.nan, "xd": np.nan, "s": np.nan, "eps": np.nan, "d_star": -1}
        for k in CONT_INFO:
            extra[f"cont_{k}"] = (np.full(n, proto[k], np.int64 if k == "d_star" else np.float64) if k in proto else np.zeros(n, bool))
        for r in ok:
            extra["traj_cont"][r["i"]] = r["traj_cont"]
            for k in CONT_INFO:
                extra[f"cont_{k}"][r["i"]] = r["cont"][k]
    if mode in FIELD_MODES:
        extra |= {k: np.full((n,) + s, fill, dt) for k, (s, dt, fill) in FIELD_ARRAYS.items()}
        for r in ok:
            if r.get("ferr"):
                ferrs.append((r["i"], r["ferr"]))
                continue
            for k in FIELD_ARRAYS:
                extra[k][r["i"]] = r["fields"][k]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".part.npz")
    np.savez_compressed(tmp, building=wb[:n], origin=wo[:n], traj=traj, level=level, peak=peak, off=off, micro=micro, stats=stats, **extra)
    os.replace(tmp, f)
    m = {"face": face, "mode": mode, "label": label_of(mode, entry), "kwargs": spec[1], "seeds": list(spec[0]) if spec[0] is not None else None,
         "windows": n, "errors": errs[:20], "error_count": len(errs), "wall_seconds": time.time() - t0, "workers": workers,
         "product_sha256": psha, "limit": limit, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if layer:
        m["continuation"] = {"constants": "ankyra_lite.continuation.CONSTANTS", "stored": "traj_cont (float32) and cont_* per window"}
    inputs_equal = _inputs_equal(ok, zx, n) if layer else None
    if mode == "ens" or mode in FIELD_MODES:                            # gate: the stored product, bit for bit
        m["gate"] = _gate(traj, level, peak, off, micro, zx, n)
        if layer:
            zc = np.load(entry["continued_path"], allow_pickle=False)
            m["gate"].update(_gate_layer(extra["traj_cont"], zc, n, inputs_equal))
            m["gate"]["pass"] = bool(m["gate"]["pass"] and m["gate"]["layer_pass"])
    if mode in FIELD_MODES:                                              # and the rebuilt quantities: the call's own fields
        st = off | micro
        fo, rb = extra["fields_ok"], extra["rebuild_ok"]
        m["fields"] = {"field_errors": ferrs[:20], "field_error_count": len(ferrs), "non_state_windows": int((~st).sum()),
                       "windows_with_fields": int(fo.sum()), "fields_on_every_non_state_window": bool(np.array_equal(fo, ~st & np.isfinite(level))),
                       "rebuilt_reproduce_the_call": int(rb.sum()), "rebuilt_mismatches": int((fo & ~rb).sum())}
        m["gate"]["fields_pass"] = bool(not ferrs and m["fields"]["fields_on_every_non_state_window"] and m["fields"]["rebuilt_mismatches"] == 0)
        m["gate"]["pass"] = bool(m["gate"]["pass"] and m["gate"]["fields_pass"])
    elif is_seed_mode(mode) and entry.get("seed_paths"):                 # gate: the stored single-seed product, bit for bit
        k = seeds_of(entry).index(int(mode[1:]))
        zs = np.load(entry["seed_paths"][k], allow_pickle=False)
        if not (np.array_equal(zs["building"][:n], wb[:n]) and np.array_equal(zs["origin"][:n], wo[:n])):
            raise SystemExit(f"{face} {mode}: the stored single-seed product has other windows")
        m["gate"] = _gate(traj, level, peak, off, micro, zs, n)
        m["gate"]["against"] = CB.show(entry["seed_paths"][k])
    elif is_group_mode(mode) and entry.get("group_paths"):               # gate: the stored three-seed group product, bit for bit
        j = int(mode[1:])
        zg = np.load(entry["group_paths"][j], allow_pickle=False)
        if not (np.array_equal(zg["building"][:n], wb[:n]) and np.array_equal(zg["origin"][:n], wo[:n])):
            raise SystemExit(f"{face} {mode}: the stored group product has other windows")
        m["gate"] = _gate(traj, level, peak, off, micro, zg, n)
        m["gate"]["against"] = CB.show(entry["group_paths"][j])
        if layer and entry.get("group_continued_paths"):
            zgc = np.load(entry["group_continued_paths"][j], allow_pickle=False)
            m["gate"].update(_gate_layer(extra["traj_cont"], zgc, n, _inputs_equal(ok, zg, n)))
            m["gate"]["pass"] = bool(m["gate"]["pass"] and m["gate"]["layer_pass"])
    m["cache_identity"] = identity["payload"]
    m["cache_identity_sha256"] = identity["sha256"]
    m["payload_sha256"] = _sha(f)
    tmp_meta = meta.with_name(meta.name + ".part")
    tmp_meta.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_meta, meta)
    complete_cache_claim(claim_path)
    return m


def load_mode(prods, face, mode, limit=None):
    """(arrays, record) of a cached mode, or (None, None); a mode identical to another returns the other's arrays."""
    label = cache_key(prods)
    entry = prods["faces"].get(face)
    if entry is None:
        return None, None
    identity = _identity_for(label, face, mode, entry, limit)
    f, meta = cache_paths(label, face, mode, limit, identity)
    m = validate_cache(f, meta, identity, allow_identical=True)
    if m is None:
        return None, None
    if m.get("identical_to"):
        z, _ = load_mode(prods, face, m["identical_to"], limit)
        return z, m
    return dict(np.load(f, allow_pickle=False)), m


def run(faces, modes=None, workers=2, prods=None, limit=None, seed_gate=8, fields=True):
    """Re-anchor the populations in the given modes (default: default_modes, and with ``fields`` the fields mode).
    ``seed_gate`` > 0: where stored single-seed products are used, seed 0 is also re-anchored on the first
    ``seed_gate`` windows and must reproduce its stored product bit for bit (one worker; cache name s0_first<n>); where
    the two stored three-seed groups are used, both groups are (g0_first<n>, g1_first<n>)."""
    prods = prods or CB.load_products()
    key = cache_key(prods)
    rec = {}
    for face in faces:
        entry = prods["faces"].get(face)
        if entry is None or not entry["npz_path"].exists():
            continue
        jobs = [(m, limit, workers) for m in (modes or default_modes(entry) + (list(FIELD_MODES) if fields else []))]
        if seed_gate and entry.get("group_paths"):
            jobs += [(f"g{j}", int(seed_gate), 1) for j in range(len(groups_of(entry))) if not any(m == f"g{j}" for m, _, _ in jobs)]
        elif seed_gate and entry.get("seed_paths") and not any(is_seed_mode(m) for m, _, _ in jobs):
            jobs.append(("s0", int(seed_gate), 1))
        for mode, lim, wk in jobs:
            if mode not in modes_for(entry):
                raise SystemExit(f"{face}: unknown mode {mode!r} for this mapping (modes: {', '.join(modes_for(entry))})")
            CB.wait_for_memory(f"re-anchor {face} {mode}")                     # ANKYRA_LITE_MIN_FREE_GB (default 5)
            t0 = time.time()
            m = run_mode(face, mode, entry, key, workers=wk, limit=lim)
            rec[f"{face}|{mode}" + ("" if lim is None else f"|first{lim}")] = m
            CB.log(f"re-anchor {face} {mode}{'' if lim is None else f' (first {lim} windows)'}: windows {m.get('windows')} "
                   f"errors {m.get('error_count')} gate {m.get('gate')} "
                   f"{m.get('identical_to') and 'identical to ' + m['identical_to'] or ''} {time.time() - t0:.0f} s")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N windows only (development, or the single-seed gate)")
    ap.add_argument("--recipe", default="R", choices=sorted(CB.RECIPE_PRODUCTS), help="products mapping (R: evaluation/products.json)")
    ap.add_argument("--products", default=None, help="another products mapping (overrides --recipe)")
    ap.add_argument("--seed-gate", type=int, default=8, help="windows of the single-seed / seed-group gate (0: none)")
    a = ap.parse_args()
    out = run([f for f in a.faces.split(",") if f], a.modes.split(",") if a.modes else None, a.workers,
              prods=CB.load_products(Path(a.products) if a.products else CB.RECIPE_PRODUCTS[a.recipe]), limit=a.limit, seed_gate=a.seed_gate)
    print(json.dumps({k: {x: v.get(x) for x in ("windows", "error_count", "gate", "wall_seconds", "identical_to")} for k, v in out.items()}, indent=1))
