# -*- coding: utf-8 -*-
"""
Cross run comparison over the local results tree

Walks the results directory for finished runs, loads the per epoch metrics, the run summary, and the
benchmark, then writes a comparison table and the training, final test, and fidelity figures into a
bare analysis folder

This reads only the on disk artifacts so it needs no cloud access, the same numbers the cloud run
summary holds are reproduced here as the dependency free ground truth
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # Render without a display
import matplotlib.pyplot as plt

MAP_KEY = re.compile(r"^map/c\d+/50_95$")  # Per class mean average precision keys
TABLE_FIELDS = ("run", "role", "method", "regime", "width", "parameters", "latency_ms", "throughput_ips", "best_map50_95", "test_map50_95")


def _read_json(path, default):
    """Read a json artifact, returning the default when it is absent"""
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label(summary, run_dir):
    """Build a readable run label from the summary role or the directory name"""
    role = summary.get("role", "")
    if role == "student":
        return f"{summary.get('method', '?')}/{summary.get('regime', '?')}"
    return role or os.path.basename(run_dir.rstrip(os.sep))


def _mean_map(scalars):
    """Mean of the per class mean average precision keys, or None when none are present"""
    values = [value for key, value in scalars.items() if MAP_KEY.match(key)]
    return sum(values) / len(values) if values else None


def discover_runs(root="results"):
    """Return every run directory under the root that holds a summary artifact"""
    summaries = glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)
    return sorted(os.path.dirname(path) for path in summaries)


def load_run(run_dir):
    """
    Load the summary, the per epoch metrics, and the benchmark for one run directory

    Args:
        run_dir: Directory holding the run artifacts

    Returns:
        A dict with the run directory, a readable label, and the loaded summary, history, benchmark
    """
    summary = _read_json(os.path.join(run_dir, "summary.json"), {})
    history = _read_json(os.path.join(run_dir, "metrics.json"), [])
    benchmark = _read_json(os.path.join(run_dir, "benchmark.json"), {})
    return {"dir": run_dir, "label": _label(summary, run_dir), "summary": summary, "history": history, "benchmark": benchmark}


def build_table(runs):
    """
    Build the comparison rows, one per run, with the headline numbers

    Args:
        runs: Loaded run dicts from load_run

    Returns:
        A list of flat row dicts keyed by the table fields
    """
    rows = []
    for run in runs:
        summary, benchmark = run["summary"], run["benchmark"]
        test_map = _mean_map(summary.get("final_test", {}))
        rows.append({
            "run": run["label"],
            "role": summary.get("role", ""),
            "method": summary.get("method", ""),
            "regime": summary.get("regime", ""),
            "width": summary.get("width", ""),
            "parameters": benchmark.get("parameters", ""),
            "latency_ms": benchmark.get("latency_ms", ""),
            "throughput_ips": benchmark.get("throughput_ips", ""),
            "best_map50_95": round(summary["best_map50_95"], 4) if "best_map50_95" in summary else "",
            "test_map50_95": round(test_map, 4) if test_map is not None else "",
        })
    return rows


def write_table(rows, out_dir):
    """Write the comparison rows as a csv and a markdown table into the analysis folder"""
    with open(os.path.join(out_dir, "comparison.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TABLE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(TABLE_FIELDS) + " |\n")
        f.write("| " + " | ".join("---" for _ in TABLE_FIELDS) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[field]) for field in TABLE_FIELDS) + " |\n")


def plot_training_curves(runs, out_dir):
    """Plot the validation mean average precision over epochs for every run"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        points = [(row["epoch"], _mean_map(row)) for row in run["history"]]
        points = [(epoch, value) for epoch, value in points if value is not None]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=run["label"])
    ax.set_title("validation map50_95 over epochs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("map50_95")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_map50_95.png"))
    plt.close(fig)


def plot_final_test(rows, out_dir):
    """Plot the final test mean average precision as a bar per run"""
    scored = [(row["run"], row["test_map50_95"]) for row in rows if row["test_map50_95"] != ""]
    if not scored:
        return
    names, values = zip(*scored)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values)
    ax.set_title("final test map50_95")
    ax.set_ylabel("map50_95")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_test_map.png"))
    plt.close(fig)


def plot_fidelity(runs, out_dir):
    """Plot the teacher student fidelity over epochs for the runs that logged it"""
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for run in runs:
        points = [(row["epoch"], row["fidelity/kl"]) for row in run["history"] if "fidelity/kl" in row]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=run["label"])
            drawn = True
    if drawn:
        ax.set_title("teacher student fidelity over epochs")
        ax.set_xlabel("epoch")
        ax.set_ylabel("softened kl")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fidelity_kl.png"))
    plt.close(fig)


def main(argv=None):
    """Aggregate the local runs into a comparison table and figures"""
    parser = argparse.ArgumentParser(description="Compare runs across the results tree")
    parser.add_argument("--results-root", dest="results_root", default="results", help="Root of the results tree")
    parser.add_argument("--out", default=None, help="Analysis output folder, defaults to results root analysis")
    args = parser.parse_args(argv)

    out_dir = args.out or os.path.join(args.results_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs = [load_run(run_dir) for run_dir in discover_runs(args.results_root)]
    rows = build_table(runs)
    write_table(rows, out_dir)
    plot_training_curves(runs, out_dir)
    plot_final_test(rows, out_dir)
    plot_fidelity(runs, out_dir)

    print(f"compared {len(runs)} runs into {out_dir}")
    for row in rows:
        print(f"  {row['run']:>24}  best {row['best_map50_95']}  test {row['test_map50_95']}")


if __name__ == "__main__":
    main()
