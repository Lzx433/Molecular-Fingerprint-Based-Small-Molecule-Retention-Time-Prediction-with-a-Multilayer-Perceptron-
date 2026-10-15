"""Audit and analyse held-out predictions for the METLIN SMRT experiment.

Run after data preparation, split generation and model training.  All primary
scores describe individual seeds; three-seed ensembles have separate outputs.
Similarity is exact against all training molecules, but only for a reproducible
sample of at most 2,000 test molecules per protocol by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import warnings

# Prefer this project's portable dependencies when present; a normal virtual
# environment remains supported when the directory is absent.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_RUNTIME = SCRIPT_ROOT / ".runtime"
if PORTABLE_RUNTIME.is_dir():
    sys.path.insert(0, str(PORTABLE_RUNTIME))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metlin_smrt_matplotlib"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTOCOLS = ("random", "scaffold")
MODELS = ("dummy", "rf", "mlp")
SEEDS = (17, 42, 73)
METRICS = ("MAE", "RMSE", "MedAE", "R2")
MODEL_LABELS = {"dummy": "Median baseline", "rf": "Random forest", "mlp": "MLP"}
COLORS = {"dummy": "#8A8A8A", "rf": "#4477AA", "mlp": "#CC6677"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--similarity-max-test", type=int, default=2000)
    parser.add_argument("--similarity-test-block", type=int, default=128)
    parser.add_argument("--similarity-train-block", type=int, default=4096)
    parser.add_argument("--similarity-threads", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for name in ("similarity_max_test", "similarity_test_block", "similarity_train_block", "dpi"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.similarity_threads <= 8:
        parser.error("--similarity-threads must be between 1 and 8")
    return args


def require_columns(frame, columns, source):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump_json(path, value):
    path.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_table(frame, directory, name):
    frame.to_csv(directory / name, index=False, float_format="%.10g")


def describe(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {
        "n": len(x), "mean": np.mean(x),
        "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
        "min": np.min(x), "q25": np.quantile(x, .25),
        "median": np.median(x), "q75": np.quantile(x, .75), "max": np.max(x),
    }


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("Metrics require nonempty aligned observations and predictions")
    error = y_pred - y_true
    sse = np.sum(error ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    r2 = (1.0 - sse / sst) if sst > 0 else (1.0 if sse == 0 else 0.0)
    return {"MAE": np.mean(np.abs(error)), "RMSE": np.sqrt(np.mean(error ** 2)),
            "MedAE": np.median(np.abs(error)), "R2": r2 if len(y_true) > 1 else np.nan}


def load_inputs(root):
    molecule_path = root / "processed" / "molecules.csv"
    molecules = pd.read_csv(molecule_path, dtype={"row_id": str, "pubchem": str,
                                                "structure_group": str, "scaffold_group": str})
    require_columns(molecules, ["row_id", "pubchem", "rt", "canonical_smiles",
                                "structure_group", "scaffold_group"], molecule_path)
    if molecules["row_id"].isna().any() or molecules["row_id"].duplicated().any():
        raise ValueError("molecules.csv row_id must be nonmissing and unique")
    for name in ("structure_group", "scaffold_group", "canonical_smiles"):
        if molecules[name].isna().any():
            raise ValueError(f"molecules.csv contains missing {name}")
    molecules["rt"] = pd.to_numeric(molecules["rt"], errors="raise")
    if not np.isfinite(molecules["rt"]).all():
        raise ValueError("molecules.csv contains nonfinite rt")
    x = np.load(root / "processed" / "X.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (len(molecules), 1024) or x.dtype != np.uint8:
        raise ValueError(f"Expected uint8 X with shape {(len(molecules), 1024)}, got {x.shape}, {x.dtype}")
    for start in range(0, len(x), 8192):
        if np.any(x[start:start + 8192] > 1):
            raise ValueError("X.npy must contain binary 0/1 fingerprints")
    indexed = molecules.set_index("row_id", drop=False)
    splits = {}
    for protocol in PROTOCOLS:
        path = root / "splits" / f"{protocol}.csv"
        split = pd.read_csv(path, dtype={"row_id": str})
        require_columns(split, ["row_id", "split"], path)
        if split["row_id"].isna().any() or split["row_id"].duplicated().any():
            raise ValueError(f"{path}: row_id must be nonmissing and unique")
        if set(split["row_id"]) != set(molecules["row_id"]):
            raise ValueError(f"{path}: row_id set does not match molecules.csv")
        split["split"] = split["split"].replace({"validation": "val", "valid": "val"})
        if set(split["split"]) != {"train", "val", "test"}:
            raise ValueError(f"{path}: expected nonempty train, val and test sets")
        # Align explicitly by identity; X row positions correspond to molecules.csv.
        splits[protocol] = split.set_index("row_id").loc[molecules["row_id"], "split"]
    return molecules, indexed, x, splits


def load_predictions(root, indexed, splits):
    pattern = re.compile(r"^(random|scaffold)_(dummy|rf|mlp)_seed(\d+)\.csv$")
    runs = {}
    audit = []
    for path in sorted((root / "outputs" / "predictions").glob("*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        protocol, model, seed_text = match.groups()
        seed = int(seed_text)
        if seed not in SEEDS:
            raise ValueError(f"Unexpected seed {seed} in {path.name}; expected {SEEDS}")
        frame = pd.read_csv(path, dtype={"row_id": str})
        require_columns(frame, ["row_id", "y_true", "y_pred"], path)
        if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
            raise ValueError(f"{path}: missing or duplicated row_id")
        expected_ids = splits[protocol].index[splits[protocol].eq("test")]
        if set(frame["row_id"]) != set(expected_ids):
            missing = len(set(expected_ids) - set(frame["row_id"]))
            extra = len(set(frame["row_id"]) - set(expected_ids))
            raise ValueError(f"{path}: test coverage mismatch ({missing} missing; {extra} extra)")
        frame = frame.set_index("row_id", drop=False).loc[expected_ids].copy()
        for name in ("y_true", "y_pred"):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            if not np.isfinite(frame[name]).all():
                raise ValueError(f"{path}: nonfinite {name}")
        if not np.allclose(frame["y_true"], indexed.loc[expected_ids, "rt"], rtol=0, atol=1e-6):
            raise ValueError(f"{path}: y_true differs from molecules.csv rt")
        runs[(protocol, model, seed)] = frame
        audit.append({"protocol": protocol, "model": model, "seed": seed,
                      "n_test": len(frame), "exact_test_coverage": True, "true_rt_verified": True,
                      "source": str(path.relative_to(root))})
    for protocol in PROTOCOLS:
        for model in MODELS:
            required = (17,) if model == "dummy" else SEEDS
            for seed in required:
                if (protocol, model, seed) not in runs:
                    raise FileNotFoundError(f"Missing required prediction file: {protocol}_{model}_seed{seed}.csv")
    return runs, audit


def read_selected_models(root, runs):
    candidates = [root / "outputs" / "selected_model.json",
                  root / "outputs" / "models" / "selected_model.json",
                  root / "outputs" / "tables" / "selected_model.json",
                  root / "models" / "selected_model.json", root / "selected_model.json"]
    selection_path = next((path for path in candidates if path.is_file()), None)
    raw = json.loads(selection_path.read_text(encoding="utf-8-sig")) if selection_path else {}
    if "selected_models" in raw:
        raw = raw["selected_models"]
    selected = {}
    for protocol in PROTOCOLS:
        entry = raw.get(protocol)
        if entry is None:
            model = "mlp"
            origin = "default; no validation selection"
            warnings.warn(f"{protocol}: selected_model.json selection unavailable; using MLP seed 17 for figures.")
        else:
            model = entry if isinstance(entry, str) else entry.get("model", entry.get("selected_model"))
            if model not in ("rf", "mlp"):
                raise ValueError(f"Invalid selected model for {protocol}: {entry}")
            origin = "validation-selected"
        if (protocol, model, 17) not in runs:
            raise ValueError(f"Seed 17 prediction missing for selected {protocol}/{model}")
        selected[protocol] = {"model": model, "seed": 17, "selection_origin": origin,
                              "metadata": entry, "source": str(selection_path.relative_to(root)) if selection_path else None}
    return selected


def metrics_and_ensembles(runs, tables):
    records = []
    for (protocol, model, seed), frame in runs.items():
        records.append({"protocol": protocol, "model": model, "seed": seed,
                        "n_test": len(frame), **score(frame["y_true"], frame["y_pred"])})
    metrics = pd.DataFrame(records).sort_values(["protocol", "model", "seed"])
    save_table(metrics, tables, "metrics.csv")
    summary_rows = []
    for (protocol, model), frame in metrics.groupby(["protocol", "model"], sort=False):
        record = {"protocol": protocol, "model": model, "n_runs": len(frame), "n_test": int(frame["n_test"].iloc[0]),
                  "seeds": ",".join(frame["seed"].astype(str))}
        for metric in METRICS:
            record[f"{metric}_mean"] = frame[metric].mean()
            record[f"{metric}_std"] = frame[metric].std(ddof=1)
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    save_table(summary, tables, "metrics_summary.csv")
    ensembles, ensemble_metrics = [], []
    for protocol in PROTOCOLS:
        for model in ("rf", "mlp"):
            base = runs[(protocol, model, 17)][["row_id", "y_true"]].copy()
            for seed in SEEDS:
                base[f"y_pred_seed{seed}"] = runs[(protocol, model, seed)].loc[base.index, "y_pred"]
            base["y_pred_ensemble"] = base[[f"y_pred_seed{s}" for s in SEEDS]].mean(axis=1)
            base.insert(0, "model", model)
            base.insert(0, "protocol", protocol)
            ensembles.append(base)
            ensemble_metrics.append({"protocol": protocol, "model": model, "seeds": "17,42,73", "n_members": 3,
                                     "n_test": len(base), **score(base["y_true"], base["y_pred_ensemble"])})
    save_table(pd.concat(ensembles, ignore_index=True), tables, "ensemble_predictions.csv")
    save_table(pd.DataFrame(ensemble_metrics), tables, "ensemble_metrics.csv")
    return metrics, summary, ensemble_metrics


def save_figure(fig, figures, stem, dpi):
    fig.savefig(figures / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def base_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.constrained_layout.use": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def plot_rt_distribution(molecules, splits, tables, figures, dpi):
    rt = molecules["rt"].to_numpy()
    edges = np.histogram_bin_edges(rt, bins=50)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(rt, bins=edges, color="#4477AA", edgecolor="white", linewidth=.25)
    axes[0].set(title=f"All processed molecules (n = {len(rt):,})", xlabel="Retention time (s)", ylabel="Molecule count")
    for j, protocol in enumerate(PROTOCOLS, 1):
        for subset, color in zip(("train", "val", "test"), ("#4477AA", "#EEAA33", "#CC6677")):
            values = rt[splits[protocol].to_numpy() == subset]
            count, _ = np.histogram(values, edges)
            density = count / (len(values) * np.diff(edges))
            axes[j].stairs(density, edges, label=f"{subset} (n = {len(values):,})", color=color, linewidth=1.5)
            for k in range(len(count)):
                rows.append({"protocol": protocol, "split": subset, "bin_left_s": edges[k],
                             "bin_right_s": edges[k + 1], "count": count[k], "density": density[k]})
        axes[j].set(title=f"{protocol.capitalize()} split", xlabel="Retention time (s)", ylabel="Probability density")
        axes[j].legend(frameon=False, fontsize=8)
    all_count, _ = np.histogram(rt, edges)
    for k, count in enumerate(all_count):
        rows.append({"protocol": "all", "split": "all", "bin_left_s": edges[k],
                     "bin_right_s": edges[k + 1], "count": count, "density": count / (len(rt) * (edges[k+1] - edges[k]))})
    save_table(pd.DataFrame(rows), tables, "rt_distribution_source.csv")
    save_figure(fig, figures, "rt_distribution", dpi)


def plot_performance(summary, figures, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        view = summary[summary["protocol"].eq(protocol)].set_index("model").loc[list(MODELS)]
        means = view["MAE_mean"].to_numpy()
        deviations = view["MAE_std"].fillna(0).to_numpy()
        bars = ax.bar(np.arange(3), means,
                      color=[COLORS[m] for m in MODELS], width=.62)
        repeated = view["n_runs"].to_numpy() > 1
        ax.errorbar(np.arange(3)[repeated], means[repeated], yerr=deviations[repeated],
                    fmt="none", ecolor="black", capsize=4, linewidth=1)
        ax.set_xticks(np.arange(3), [MODEL_LABELS[m] for m in MODELS])
        ax.set(title=f"{protocol.capitalize()} split", ylabel="Test MAE (s)")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        for bar, mean, dev, n_runs in zip(bars, means, deviations, view["n_runs"]):
            label = f"{mean:.2f}" if n_runs == 1 else f"{mean:.2f} ± {dev:.2f}"
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, mean + dev), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=.2)
    fig.suptitle("Individual runs: mean ± sample SD across seeds\nSingle-run baseline has no estimated seed SD", fontsize=11)
    save_figure(fig, figures, "mae_by_protocol", dpi)


def representative_frame(runs, selected):
    frames = []
    for protocol in PROTOCOLS:
        choice = selected[protocol]
        frame = runs[(protocol, choice["model"], 17)].copy()
        frame.insert(0, "protocol", protocol)
        frame.insert(1, "model", choice["model"])
        frame.insert(2, "seed", 17)
        frame["residual_s"] = frame["y_pred"] - frame["y_true"]
        frame["absolute_error_s"] = frame["residual_s"].abs()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def choice_title(protocol, selected):
    choice = selected[protocol]
    return f"{protocol.capitalize()}: {MODEL_LABELS[choice['model']]}, seed 17\n{choice['selection_origin']}"


def plot_prediction_and_residuals(representative, selected, tables, figures, dpi):
    save_table(representative, tables, "representative_predictions_source.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, protocol in zip(axes, PROTOCOLS):
        view = representative[representative["protocol"].eq(protocol)]
        hb = ax.hexbin(view["y_true"], view["y_pred"], gridsize=55, mincnt=1, bins="log", cmap="viridis", rasterized=True)
        lo = min(view["y_true"].min(), view["y_pred"].min())
        hi = max(view["y_true"].max(), view["y_pred"].max())
        pad = max((hi - lo) * .03, 1)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#BB5566", linewidth=1)
        ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), xlabel="Observed retention time (s)",
               ylabel="Predicted retention time (s)", title=choice_title(protocol, selected))
        ax.set_aspect("equal", adjustable="box")
        values = score(view["y_true"], view["y_pred"])
        ax.text(.04, .96, f"n = {len(view):,}\nMAE = {values['MAE']:.1f} s\nR² = {values['R2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        fig.colorbar(hb, ax=ax, label="Molecules per hexagon (log scale)", shrink=.8)
    save_figure(fig, figures, "predicted_vs_observed", dpi)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    residual_rows = []
    for j, protocol in enumerate(PROTOCOLS):
        view = representative[representative["protocol"].eq(protocol)]
        ax = axes[0, j]
        hb = ax.hexbin(view["y_true"], view["residual_s"], gridsize=55, mincnt=1, bins="log", cmap="viridis", rasterized=True)
        ax.axhline(0, color="#BB5566", linestyle="--", linewidth=1)
        ax.set(xlabel="Observed retention time (s)", ylabel="Prediction − observation (s)", title=choice_title(protocol, selected))
        fig.colorbar(hb, ax=ax, label="Count (log scale)", shrink=.85)
        counts, edges, _ = axes[1, j].hist(view["residual_s"], bins=50, color=COLORS[selected[protocol]["model"]], edgecolor="white", linewidth=.3)
        axes[1, j].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[1, j].set(xlabel="Prediction − observation (s)", ylabel="Molecule count")
        for k, count in enumerate(counts):
            residual_rows.append({"protocol": protocol, "model": selected[protocol]["model"], "seed": 17,
                                  "bin_left_s": edges[k], "bin_right_s": edges[k+1], "count": int(count)})
    save_table(pd.DataFrame(residual_rows), tables, "residual_histogram_source.csv")
    save_figure(fig, figures, "residuals", dpi)


def quartile_analysis(molecules, splits, runs, tables, figures, dpi):
    records, edges_records = [], []
    rt = molecules["rt"].to_numpy()
    for protocol in PROTOCOLS:
        training_rt = rt[splits[protocol].to_numpy() == "train"]
        edges = np.quantile(training_rt, [0, .25, .5, .75, 1])
        edges_records.append({"protocol": protocol, **{name: value for name, value in zip(("train_min", "q25", "q50", "q75", "train_max"), edges)}})
        for model in MODELS:
            view = runs[(protocol, model, 17)]
            groups = np.searchsorted(edges[1:4], view["y_true"].to_numpy(), side="right")
            for k in range(4):
                part = view.iloc[np.flatnonzero(groups == k)]
                values = score(part["y_true"], part["y_pred"]) if len(part) else {m: np.nan for m in METRICS}
                records.append({"protocol": protocol, "model": model, "seed": 17, "rt_quartile": k+1,
                                "n_test": len(part), "lower_bound_s": -np.inf if k == 0 else edges[k],
                                "upper_bound_s": np.inf if k == 3 else edges[k+1], **values})
    frame = pd.DataFrame(records)
    save_table(frame, tables, "rt_quartile_metrics_source.csv")
    save_table(pd.DataFrame(edges_records), tables, "training_rt_quartiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        for model in MODELS:
            part = frame[frame["protocol"].eq(protocol) & frame["model"].eq(model)]
            ax.plot(part["rt_quartile"], part["MAE"], marker="o", color=COLORS[model], label=MODEL_LABELS[model])
        ax.set(xticks=[1,2,3,4], xticklabels=["Q1", "Q2", "Q3", "Q4"], xlabel="RT bins defined by training-set quartiles",
               ylabel="Test MAE (s)", title=f"{protocol.capitalize()} split | seed 17")
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
    save_figure(fig, figures, "mae_by_rt_quartile", dpi)
    return {"boundaries": edges_records, "metrics": frame.to_dict(orient="records"),
            "tie_rule": "Values equal to an internal cut are assigned to the upper bin.",
            "outer_bins": "Q1 and Q4 extend to negative and positive infinity to include all test RTs.",
            "repeated_boundaries": "Repeated training quartiles may leave empty bins; no data-dependent rebucketing is performed."}


def nearest_train_similarity(x, train_positions, test_positions, test_block, train_block, threads):
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=threads, user_api="blas")
    except ImportError:
        raise RuntimeError("Install threadpoolctl to enforce the BLAS thread limit for similarity analysis")
    result = np.zeros(len(test_positions), dtype=np.float32)
    test_sums = np.asarray(x[test_positions].sum(axis=1), dtype=np.float32)
    train_sums = np.asarray(x[train_positions].sum(axis=1), dtype=np.float32)
    with context:
        for start in range(0, len(test_positions), test_block):
            stop = min(start+test_block, len(test_positions))
            left = np.asarray(x[test_positions[start:stop]], dtype=np.float32)
            best = np.zeros(stop-start, dtype=np.float32)
            for train_start in range(0, len(train_positions), train_block):
                train_stop = min(train_start+train_block, len(train_positions))
                right = np.asarray(x[train_positions[train_start:train_stop]], dtype=np.float32)
                intersection = left @ right.T
                union = test_sums[start:stop, None] + train_sums[None, train_start:train_stop] - intersection
                # Tanimoto(0, 0)=1; all other zero-vs-nonzero similarities are 0.
                np.divide(intersection, union, out=intersection, where=union > 0)
                intersection[union == 0] = 1.0
                np.maximum(best, intersection.max(axis=1), out=best)
            result[start:stop] = best
            print(f"  nearest-training Tanimoto: {stop}/{len(test_positions)} test molecules", flush=True)
    return np.clip(result, 0, 1)


def similarity_analysis(args, molecules, x, splits, representative, selected, tables, figures):
    source_rows, binned_rows, summary = [], [], {}
    position = pd.Series(np.arange(len(molecules)), index=molecules["row_id"])
    # Fixed edges are chosen before seeing either predictions or similarities.
    edges = np.array([0, .2, .4, .6, .8, 1.0])
    for protocol in PROTOCOLS:
        split = splits[protocol]
        train_ids = split.index[split.eq("train")]
        test_ids = split.index[split.eq("test")]
        rng = np.random.default_rng(2026)
        test_pick = np.sort(rng.choice(len(test_ids), size=min(args.similarity_max_test, len(test_ids)), replace=False))
        chosen_ids = test_ids[test_pick]
        print(f"{protocol}: exact nearest similarity for {len(chosen_ids)}/{len(test_ids)} test molecules against {len(train_ids)} training molecules", flush=True)
        values = nearest_train_similarity(x, position.loc[train_ids].to_numpy(), position.loc[chosen_ids].to_numpy(),
                                          args.similarity_test_block, args.similarity_train_block, args.similarity_threads)
        chosen_predictions = representative[representative["protocol"].eq(protocol)].set_index("row_id").loc[chosen_ids]
        view = chosen_predictions.reset_index().copy()
        view["max_train_tanimoto"] = values
        source_rows.append(view)
        groups = np.minimum(np.searchsorted(edges, values, side="right") - 1, len(edges)-2)
        for k in range(len(edges)-1):
            part = view.iloc[np.flatnonzero(groups == k)]
            binned_rows.append({"protocol": protocol, "model": selected[protocol]["model"], "seed": 17,
                               "similarity_bin_left": edges[k], "similarity_bin_right": edges[k+1],
                               "n_sampled_test": len(part), "MAE": part["absolute_error_s"].mean(),
                               "MedAE": part["absolute_error_s"].median()})
        similarity_ranks = pd.Series(values).rank()
        error_ranks = view["absolute_error_s"].reset_index(drop=True).rank()
        rank_correlation = (similarity_ranks.corr(error_ranks)
                            if similarity_ranks.nunique() > 1 and error_ranks.nunique() > 1 else np.nan)
        summary[protocol] = {"n_training_compared": len(train_ids), "n_test_total": len(test_ids),
                             "n_test_sampled": len(chosen_ids), "test_sample_fraction": len(chosen_ids)/len(test_ids),
                             "sampling_seed": 2026, "sampling": "uniform without replacement; no target or error stratification",
                             "nearest_neighbor_search": "exact maximum over every training fingerprint for each sampled test molecule",
                             "max_train_tanimoto": describe(values),
                             "sample_absolute_error_s": describe(view["absolute_error_s"]),
                             "spearman_similarity_absolute_error": rank_correlation,
                             "n_tanimoto_one_in_sample": int(np.sum(values == 1)),
                             "binned_error_statistics": [row for row in binned_rows if row["protocol"] == protocol],
                             "duplicate_interpretation": "Tanimoto=1 identifies an identical 1024-bit fingerprint, not necessarily an identical molecule; bit collisions are possible."}
    sources = pd.concat(source_rows, ignore_index=True)
    bins = pd.DataFrame(binned_rows)
    save_table(sources, tables, "max_train_tanimoto_sample.csv")
    save_table(bins, tables, "similarity_error_bins_source.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, protocol in enumerate(PROTOCOLS):
        view = sources[sources["protocol"].eq(protocol)]
        hb = axes[0,j].hexbin(view["max_train_tanimoto"], view["absolute_error_s"], gridsize=40, mincnt=1,
                             bins="log", cmap="viridis", rasterized=True)
        axes[0,j].set(xlim=(0,1), xlabel="Maximum training-set Tanimoto similarity", ylabel="Absolute error (s)",
                      title=f"{choice_title(protocol, selected)}\nUniform test subset: n = {len(view):,}")
        fig.colorbar(hb, ax=axes[0,j], label="Sampled molecules (log scale)", shrink=.85)
        b = bins[bins["protocol"].eq(protocol)]
        midpoints = (b["similarity_bin_left"] + b["similarity_bin_right"]) / 2
        axes[1,j].plot(midpoints, b["MAE"], "o-", color=COLORS[selected[protocol]["model"]])
        for center, mae, count in zip(midpoints, b["MAE"], b["n_sampled_test"]):
            if np.isfinite(mae):
                axes[1,j].annotate(f"n={count}", (center, mae), xytext=(0,7), textcoords="offset points", ha="center", fontsize=8)
        axes[1,j].set(xlim=(0,1), xlabel="Maximum training-set Tanimoto similarity (binned)", ylabel="Sample MAE (s)")
        axes[1,j].grid(alpha=.18)
        axes[1,j].margins(y=.2)
    save_figure(fig, figures, "similarity_vs_error", args.dpi)
    return summary


def data_summary(molecules, x, splits):
    groups = molecules["structure_group"].value_counts()
    # Identical hashed fingerprints are an exact bit-pattern count, not a count
    # of molecular duplicates. Do not substitute this for structure grouping.
    packed = np.packbits(x, axis=1)
    _, fingerprint_counts = np.unique(packed, axis=0, return_counts=True)
    output = {"n_molecules": len(molecules), "n_features": x.shape[1], "feature_dtype": str(x.dtype),
              "feature_values": [0, 1], "rt_unit": "seconds", "rt": describe(molecules["rt"]),
              "n_unique_pubchem": molecules["pubchem"].nunique(),
              "n_unique_structure_groups": len(groups), "n_repeated_structure_groups": int((groups > 1).sum()),
              "n_rows_in_repeated_structure_groups": int(groups[groups > 1].sum()),
              "n_unique_scaffold_groups": molecules["scaffold_group"].nunique(),
              "fingerprint_pattern_counts": {"n_unique_patterns": len(fingerprint_counts),
                                              "n_repeated_patterns": int((fingerprint_counts > 1).sum()),
                                              "n_rows_in_repeated_patterns": int(fingerprint_counts[fingerprint_counts > 1].sum()),
                                              "interpretation": "Exact matches of the provided 1024-bit representation; not proof of molecular identity."},
              "split_statistics": {}}
    indexed = molecules.set_index("row_id")
    for protocol, split in splits.items():
        part = {name: {"n": int(split.eq(name).sum()), "rt": describe(indexed.loc[split.index[split.eq(name)], "rt"])}
                for name in ("train", "val", "test")}
        overlaps = {}
        for grouping in ("structure_group", "scaffold_group"):
            subsets = {name: set(indexed.loc[split.index[split.eq(name)], grouping]) for name in ("train", "val", "test")}
            overlaps[grouping] = {f"{a}__{b}": len(subsets[a] & subsets[b])
                                  for a,b in (("train","val"),("train","test"),("val","test"))}
        part["group_overlap_counts"] = overlaps
        output["split_statistics"][protocol] = part
    return output


def fingerprint_match_sensitivity(molecules, x, splits, runs, tables):
    """An input-defined sensitivity check; never changes the primary test set."""
    packed = np.packbits(x, axis=1)
    records, flags, details = [], [], {}
    for protocol in PROTOCOLS:
        labels = splits[protocol].to_numpy()
        train_positions = np.flatnonzero(labels == "train")
        test_positions = np.flatnonzero(labels == "test")
        train_patterns = {row.tobytes() for row in packed[train_positions]}
        test_patterns = [row.tobytes() for row in packed[test_positions]]
        matched = np.array([pattern in train_patterns for pattern in test_patterns], dtype=bool)
        test_ids = molecules.iloc[test_positions]["row_id"].to_numpy()
        excluded_ids = set(test_ids[matched])
        details[protocol] = {
            "n_training": len(train_positions), "n_test_full": len(test_positions),
            "n_test_with_identical_training_fingerprint": int(matched.sum()),
            "n_matching_fingerprint_patterns": len({test_patterns[i] for i in np.flatnonzero(matched)}),
            "n_test_remaining": int((~matched).sum()), "excluded_fraction": float(matched.mean()),
        }
        for row_id, matched_train in zip(test_ids, matched):
            flags.append({"protocol": protocol, "row_id": row_id, "identical_train_fingerprint": bool(matched_train)})
        for model in MODELS:
            full = runs[(protocol, model, 17)]
            remaining = full.loc[~full["row_id"].isin(excluded_ids)]
            full_score = score(full["y_true"], full["y_pred"])
            filtered_score = score(remaining["y_true"], remaining["y_pred"]) if len(remaining) else {m: np.nan for m in METRICS}
            record = {"protocol": protocol, "model": model, "seed": 17,
                      "n_test_full": len(full), "n_excluded_identical_train_fp": len(full)-len(remaining),
                      "n_test_remaining": len(remaining), "excluded_fraction": (len(full)-len(remaining))/len(full),
                      **filtered_score}
            for metric in METRICS:
                record[f"full_{metric}"] = full_score[metric]
                record[f"delta_{metric}_vs_full"] = filtered_score[metric] - full_score[metric]
            records.append(record)
    save_table(pd.DataFrame(records), tables, "sensitivity_no_train_fp_match.csv")
    save_table(pd.DataFrame(flags), tables, "fingerprint_train_test_overlap.csv")
    return {"protocols": details, "metrics": records,
            "definition": "Exclude a test row only if its complete 1024-bit fingerprint equals that of any training row; packbits byte strings are compared exactly.",
            "scope": "Sensitivity analysis for seed 17 only. Primary metrics retain the entire fixed test set; no split or model is changed.",
            "interpretation": "Equal fingerprints do not prove equal molecules. The exclusion rule depends only on inputs, never on observed prediction errors."}


def split_composition(molecules, splits, tables):
    """Measure scaffold frequency in all processed rows before subsetting.

    A global singleton is a scaffold represented by exactly one molecule in the
    complete processed data. It is not a scaffold that merely happens to occur
    once in a particular subset; that separate definition is also tabulated.
    """
    global_scaffold_sizes = molecules["scaffold_group"].value_counts()
    indexed = molecules.set_index("row_id", drop=False)
    records = []
    for protocol, labels in splits.items():
        for subset in ("train", "val", "test"):
            ids = labels.index[labels.eq(subset)]
            view = indexed.loc[ids]
            global_sizes = view["scaffold_group"].map(global_scaffold_sizes)
            local_sizes = view["scaffold_group"].value_counts()
            local_row_sizes = view["scaffold_group"].map(local_sizes)
            n = len(view)
            n_singletons_global = int(global_sizes.eq(1).sum())
            n_singletons_local = int(local_row_sizes.eq(1).sum())
            n_large_group_rows = int(global_sizes.ge(3).sum())
            n_early = int(view["rt"].le(300).sum())
            records.append({
                "protocol": protocol, "split": subset, "n_rows": n,
                "n_structure_groups": int(view["structure_group"].nunique()),
                "n_scaffold_groups": int(view["scaffold_group"].nunique()),
                "rt_mean_s": float(view["rt"].mean()),
                "rt_median_s": float(view["rt"].median()),
                "scaffold_group_max_size_global": int(global_sizes.max()),
                "scaffold_group_max_size_within_split": int(local_sizes.max()),
                "singleton_scaffold_rows_global": n_singletons_global,
                "singleton_scaffold_fraction_global": n_singletons_global / n,
                "singleton_scaffold_rows_within_split": n_singletons_local,
                "singleton_scaffold_fraction_within_split": n_singletons_local / n,
                "scaffold_rows_global_group_size_ge3": n_large_group_rows,
                "scaffold_fraction_global_group_size_ge3": n_large_group_rows / n,
                "rt_le_300_rows": n_early, "rt_le_300_fraction": n_early / n,
            })
    frame = pd.DataFrame(records)
    save_table(frame, tables, "split_composition.csv")
    return {
        "global_frequency_source": "All rows in processed/molecules.csv, before selecting any protocol or subset.",
        "n_processed_rows": len(molecules), "n_global_scaffold_groups": len(global_scaffold_sizes),
        "global_definition": "Fields ending in _global use scaffold group sizes in the complete processed dataset, even when the random split distributes a scaffold across subsets.",
        "within_split_definition": "Fields ending in _within_split recount scaffold sizes only in the indicated protocol/subset.",
        "fraction_denominator": "n_rows of the indicated protocol/subset; fractions are on the 0–1 scale.",
        "rows": records,
    }


def prediction_range_diagnostics(molecules, splits, runs, tables):
    """Describe raw prediction ranges without clipping or removing any row."""
    rt = molecules.set_index("row_id")["rt"]
    training_maxima = {protocol: float(rt.loc[labels.index[labels.eq("train")]].max())
                       for protocol, labels in splits.items()}
    records = []
    for (protocol, model, seed), frame in sorted(runs.items()):
        values = frame["y_pred"].to_numpy()
        n = len(values)
        n_negative = int(np.sum(values < 0))
        n_above = int(np.sum(values > training_maxima[protocol]))
        records.append({"protocol": protocol, "model": model, "seed": seed,
                        "n_test": n, "min_y_pred": float(values.min()), "max_y_pred": float(values.max()),
                        "n_negative": n_negative, "fraction_negative": n_negative / n,
                        "training_rt_max_s": training_maxima[protocol],
                        "n_above_training_rt_max": n_above, "fraction_above_training_rt_max": n_above / n})
    save_table(pd.DataFrame(records), tables, "prediction_ranges.csv")
    return {"rows": records, "prediction_units": "seconds",
            "negative_definition": "Raw y_pred < 0; these values are physically nonnegative-target violations.",
            "above_training_max_definition": "Raw y_pred exceeds the maximum observed training RT of its protocol; this is a range diagnostic, not proof that the prediction is physically impossible.",
            "action": "No clipping, exclusions, retraining, reselection, or changes to reported metrics."}


def rt_300s_diagnostic(runs, tables):
    """Exploratory RT strata for seed 17, retaining raw predictions and all rows."""
    records = []
    for protocol in PROTOCOLS:
        for model in MODELS:
            full = runs[(protocol, model, 17)]
            strata = (("rt_le_300s", full["y_true"].le(300)),
                      ("rt_gt_300s", full["y_true"].gt(300)))
            for label, mask in strata:
                view = full.loc[mask]
                residual = view["y_pred"] - view["y_true"]
                values = score(view["y_true"], view["y_pred"]) if len(view) else {m: np.nan for m in METRICS}
                records.append({"protocol": protocol, "model": model, "seed": 17,
                                "rt_stratum": label, "cutoff_s": 300,
                                "n_test": len(view), "fraction_of_full_test": len(view)/len(full),
                                **values, "mean_residual_s": residual.mean(), "median_residual_s": residual.median()})
    save_table(pd.DataFrame(records), tables, "rt_300s_diagnostic.csv")
    return {"rows": records,
            "threshold_interpretation": "300 seconds is an exploratory descriptive threshold, not a verified chromatographic dead time.",
            "scope": "Seed 17 for each model and protocol; strata are defined by observed test RT. No sample is removed from primary evaluation.",
            "action": "No model selection, preprocessing, clipping, filtering, or primary metric changes depend on this diagnostic.",
            "r2_caution": "Within-stratum R2 uses only that stratum's target variance and may be highly negative in a narrow RT range; do not compare it directly with full-test R2."}


def main():
    args = parse_args()
    root = args.project_root.resolve()
    tables, figures = root / "outputs" / "tables", root / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    base_style()
    print("Loading and auditing processed data, fixed splits, and all model predictions...", flush=True)
    molecules, indexed, x, splits = load_inputs(root)
    runs, audit = load_predictions(root, indexed, splits)
    selected = read_selected_models(root, runs)
    metrics, summary, ensemble_scores = metrics_and_ensembles(runs, tables)
    representative = representative_frame(runs, selected)
    print("Writing descriptive statistics and figures...", flush=True)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "project_root": str(root), "data": data_summary(molecules, x, splits),
              "prediction_audit": audit, "representative_models": selected,
              "primary_metrics": metrics.to_dict(orient="records"),
              "primary_metrics_summary": summary.to_dict(orient="records"),
              "primary_uncertainty": "Mean and sample SD across training seeds on the same fixed split; not confidence intervals or repeated-split uncertainty. Single-run SD is null.",
              "ensemble_metrics": ensemble_scores,
              "ensemble_interpretation": "Separate arithmetic mean of seed 17, 42, 73 predictions; not included in primary seed-averaged metrics.",
              "representative_residuals": {p: describe(representative.loc[representative["protocol"].eq(p), "residual_s"]) for p in PROTOCOLS}}
    report["fingerprint_match_sensitivity"] = fingerprint_match_sensitivity(molecules, x, splits, runs, tables)
    report["split_composition"] = split_composition(molecules, splits, tables)
    report["prediction_ranges"] = prediction_range_diagnostics(molecules, splits, runs, tables)
    report["rt_300s_diagnostic"] = rt_300s_diagnostic(runs, tables)
    plot_rt_distribution(molecules, splits, tables, figures, args.dpi)
    plot_performance(summary, figures, args.dpi)
    plot_prediction_and_residuals(representative, selected, tables, figures, args.dpi)
    report["rt_quartile_analysis"] = quartile_analysis(molecules, splits, runs, tables, figures, args.dpi)
    report["similarity_analysis"] = {"status": "pending"}
    dump_json(tables / "analysis_summary.json", report)
    if args.skip_similarity:
        report["similarity_analysis"] = {"status": "skipped by --skip-similarity", "n_test_evaluated": 0,
                                         "note": "Any existing similarity output files are from an earlier run; this run does not update them."}
    else:
        report["similarity_analysis"] = {"status": "completed", "protocols": similarity_analysis(
            args, molecules, x, splits, representative, selected, tables, figures)}
    dump_json(tables / "analysis_summary.json", report)
    print(f"Complete. Tables: {tables}\nFigures: {figures}", flush=True)


if __name__ == "__main__":
    main()
