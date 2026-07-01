#!/usr/bin/env python3
"""make_manifest.py — build a provenance manifest for one campaign run.

Import and call `write_manifest(...)` from inside a run script, or from the
subagent that trains one grid cell. It captures git state, data-file
checksums, and the environment hash so the number is reproducible later.

Example (inside a run script, after computing r2/rmse):
    from make_manifest import write_manifest
    write_manifest(
        out_dir="results/campaign_dg_bakeoff",
        run_id="rf_seed0",
        script=__file__,
        model="RandomForest", feature_set="metal+cond+ligand",
        hyperparams={"n_estimators": 600, "max_depth": None},
        split_protocol="molecule-grouped-5fold", seed=0,
        data_files=["data/Training_Data_V27.csv", "data/Testing_Data_V39.csv"],
        metric_of_record="signed_R2", r2=0.461, rmse=6.31, n=2273,
    )
"""
import os, json, time, hashlib, subprocess, sys


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _env_hash():
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                         text=True, stderr=subprocess.DEVNULL)
        return hashlib.sha256(freeze.encode()).hexdigest()
    except Exception:
        return None


def write_manifest(out_dir, run_id, script, model, feature_set, hyperparams,
                   split_protocol, seed, data_files, metric_of_record,
                   r2, rmse, n, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    manifest = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": sha,
        "git_dirty": dirty,
        "script": os.path.relpath(script) if script else None,
        "model": model,
        "feature_set": feature_set,
        "hyperparams": hyperparams,
        "split_protocol": split_protocol,
        "seed": seed,
        "data_files": [
            {"path": p, "sha256": _sha256(p) if os.path.exists(p) else None}
            for p in data_files
        ],
        "env_hash": _env_hash(),
        "metric_of_record": metric_of_record,
        "r2": r2,
        "rmse": rmse,
        "n": n,
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(out_dir, f"run_{run_id}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    if dirty:
        print(f"WARNING: git working tree is dirty — run_{run_id} is not "
              f"cleanly reproducible until committed.")
    print(f"wrote {path}")
    return path
