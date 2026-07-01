#!/usr/bin/env python3
"""aggregate_campaign.py — roll up per-run manifests into a master table.

Reads every run_*.json in a campaign directory and writes:
  - master.csv        one row per run
  - master_rollup.csv mean +/- std across seeds, per (model, feature_set),
                      ranked by the metric of record, with a tie flag
                      (lead <= seed-to-seed std => call it a tie).

Usage:
    python3 aggregate_campaign.py results/campaign_<name>
"""
import os, sys, json, glob, statistics as st


def load_runs(campaign_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(campaign_dir, "run_*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def main(campaign_dir):
    runs = load_runs(campaign_dir)
    if not runs:
        print(f"no run_*.json manifests found in {campaign_dir}")
        sys.exit(1)

    # master.csv: flat, one row per run
    cols = ["run_id", "model", "feature_set", "seed", "split_protocol",
            "metric_of_record", "r2", "rmse", "n", "git_sha", "git_dirty"]
    master = os.path.join(campaign_dir, "master.csv")
    with open(master, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in runs:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # rollup: group by (model, feature_set), aggregate the metric of record
    groups = {}
    for r in runs:
        key = (r.get("model"), r.get("feature_set"))
        groups.setdefault(key, []).append(r)

    def metric_vals(rs):
        mor = rs[0].get("metric_of_record", "r2")
        # metric of record is stored under r2 by convention unless it's rmse
        field = "rmse" if str(mor).lower().startswith("rmse") else "r2"
        return field, [float(x[field]) for x in rs if x.get(field) not in (None, "")]

    summary = []
    for (model, fs), rs in groups.items():
        field, vals = metric_vals(rs)
        if not vals:
            continue
        mean = st.mean(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        any_dirty = any(x.get("git_dirty") for x in rs)
        summary.append({"model": model, "feature_set": fs, "metric": field,
                        "mean": mean, "std": sd, "n_seeds": len(vals),
                        "any_dirty": any_dirty})

    # rank: higher is better for r2, lower is better for rmse
    field = summary[0]["metric"] if summary else "r2"
    reverse = not field.startswith("rmse")
    summary.sort(key=lambda d: d["mean"], reverse=reverse)

    # tie flag: is the winner's lead over #2 within the winner's own seed std?
    for i, d in enumerate(summary):
        if i == 0 and len(summary) > 1:
            lead = abs(summary[0]["mean"] - summary[1]["mean"])
            d["tie_with_next"] = lead <= summary[0]["std"]
        else:
            d["tie_with_next"] = ""

    rollup = os.path.join(campaign_dir, "master_rollup.csv")
    rcols = ["rank", "model", "feature_set", "metric", "mean", "std",
             "n_seeds", "tie_with_next", "any_dirty"]
    with open(rollup, "w") as f:
        f.write(",".join(rcols) + "\n")
        for i, d in enumerate(summary):
            f.write(",".join(str(v) for v in [
                i + 1, d["model"], d["feature_set"], d["metric"],
                f"{d['mean']:.4f}", f"{d['std']:.4f}", d["n_seeds"],
                d["tie_with_next"], d["any_dirty"]]) + "\n")

    print(f"wrote {master} ({len(runs)} runs)")
    print(f"wrote {rollup} ({len(summary)} model/feature groups)")
    if summary:
        w = summary[0]
        tie = " (TIE — prefer the simpler/faster model)" if w.get("tie_with_next") else ""
        print(f"winner: {w['model']} / {w['feature_set']} — "
              f"{w['metric']} {w['mean']:.4f} +/- {w['std']:.4f}{tie}")
        if any(d["any_dirty"] for d in summary):
            print("WARNING: some runs had a dirty git tree — re-run clean "
                  "before quoting a headline number.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 aggregate_campaign.py results/campaign_<name>")
        sys.exit(1)
    main(sys.argv[1])
