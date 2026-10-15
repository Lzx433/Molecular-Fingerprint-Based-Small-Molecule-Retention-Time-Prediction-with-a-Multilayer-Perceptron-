"""Training-only chemical description and an audit of excluded-label distributions.

This is supplementary descriptive analysis. It never changes the fingerprint
matrix, molecule retention policy, train/validation/test splits, or trained models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[_thread_variable] = "4"

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime"))
_mpl_cache = ROOT / "outputs" / ".matplotlib_chemical_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


LOG = logging.getLogger("smrt.chemical_analysis")
DESCRIPTOR_FUNCTIONS = {
    "MW": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "RotatableBonds": Lipinski.NumRotatableBonds,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}
DESCRIPTOR_DEFINITIONS = {
    "MW": "RDKit Descriptors.MolWt; average molecular weight, g/mol",
    "MolLogP": "RDKit Crippen.MolLogP; computed Wildman-Crippen logP, dimensionless",
    "TPSA": "RDKit rdMolDescriptors.CalcTPSA; default options, square angstroms",
    "HBD": "RDKit Lipinski.NumHDonors; hydrogen-bond donor count",
    "HBA": "RDKit Lipinski.NumHAcceptors; hydrogen-bond acceptor count",
    "RotatableBonds": "RDKit Lipinski.NumRotatableBonds; RDKit default definition",
    "HeavyAtomCount": "RDKit Descriptors.HeavyAtomCount; non-hydrogen atom count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_training_records(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecules = pd.read_csv(
        project_root / "processed" / "molecules.csv", dtype={"pubchem": "string"}
    )
    splits = pd.read_csv(project_root / "splits" / "random.csv")
    required = {"row_id", "pubchem", "rt", "canonical_smiles"}
    if not required.issubset(molecules.columns):
        raise ValueError(f"molecules.csv is missing columns: {required - set(molecules.columns)}")
    if molecules.row_id.duplicated().any() or splits.row_id.duplicated().any():
        raise ValueError("row_id must be unique in molecules.csv and random.csv")
    if set(molecules.row_id) != set(splits.row_id):
        raise ValueError("Molecule and split row_id coverage differs")
    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("Unexpected split labels")
    training_ids = splits.loc[splits["split"] == "train", ["row_id"]]
    training = training_ids.merge(
        molecules[list(required)], on="row_id", how="left", validate="one_to_one"
    ).sort_values("row_id").reset_index(drop=True)
    if training.empty or training[list(required)].isna().any().any():
        raise ValueError("Missing required values in training rows")
    if not np.isfinite(training.rt.to_numpy(dtype=float)).all():
        raise ValueError("Training RT values must be finite")
    LOG.info("Selected %d training rows; validation/test structures are not analyzed", len(training))
    return training, molecules


def compute_descriptors(training: pd.DataFrame) -> pd.DataFrame:
    records = []
    for index, row in enumerate(training.itertuples(index=False), start=1):
        mol = Chem.MolFromSmiles(row.canonical_smiles, sanitize=True)
        if mol is None:
            raise ValueError(
                f"Canonical SMILES could not be parsed for row_id={row.row_id}, CID={row.pubchem}"
            )
        values = {name: function(mol) for name, function in DESCRIPTOR_FUNCTIONS.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite descriptor for row_id={row.row_id}")
        records.append({
            "row_id": row.row_id,
            "pubchem": row.pubchem,
            "rt": row.rt,
            **values,
        })
        if index % 10000 == 0:
            LOG.info("Computed descriptors for %d / %d training molecules", index, len(training))
    result = pd.DataFrame(records)
    if result.row_id.tolist() != training.row_id.tolist():
        raise AssertionError("Descriptor output row alignment failed")
    return result


def correlations(descriptors: pd.DataFrame) -> pd.DataFrame:
    records = []
    subsets = {
        "all_random_training": descriptors,
        "random_training_rt_gt_300s": descriptors.loc[descriptors.rt > 300],
    }
    for subset_name, subset in subsets.items():
        for descriptor_name in DESCRIPTOR_FUNCTIONS:
            if len(subset) < 3 or subset[descriptor_name].nunique() < 2 or subset.rt.nunique() < 2:
                raise ValueError(f"Cannot calculate Spearman correlation for {subset_name}/{descriptor_name}")
            rho = float(spearmanr(subset[descriptor_name], subset.rt, nan_policy="raise").statistic)
            records.append({
                "subset": subset_name,
                "n": len(subset),
                "descriptor": descriptor_name,
                "spearman_rho": rho,
                "association_not_causation": True,
                "inferential_tests_reported": False,
                "multiplicity_adjustment_performed": False,
            })
    return pd.DataFrame(records)


def exclusion_summary(project_root: Path, retained: pd.DataFrame) -> pd.DataFrame:
    excluded = pd.read_csv(project_root / "processed" / "excluded_records.csv")
    original_n = len(retained) + len(excluded)
    records = []
    for cohort, frame in (("retained_all_partitions", retained), ("excluded_structure_parse", excluded)):
        values = frame.rt.to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Empty or invalid RT values in {cohort}")
        quantiles = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        records.append({
            "cohort": cohort,
            "n": len(values),
            "percent_of_original_rows": 100 * len(values) / original_n,
            "rt_mean_s": float(np.mean(values)),
            "rt_std_sample_s": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            **{name: float(value) for name, value in zip(
                ("rt_min_s", "rt_p05_s", "rt_q25_s", "rt_median_s", "rt_q75_s", "rt_p95_s", "rt_max_s"),
                quantiles,
            )},
            "n_rt_le_300s": int(np.count_nonzero(values <= 300)),
            "percent_rt_le_300s": float(100 * np.mean(values <= 300)),
            "n_rt_le_500s": int(np.count_nonzero(values <= 500)),
            "percent_rt_le_500s": float(100 * np.mean(values <= 500)),
        })
    return pd.DataFrame(records)


def draw_figure(descriptors: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), sharey=True, layout="constrained")
    definitions = (
        ("MW", "Molecular weight (g/mol)", "A  Molecular weight"),
        ("MolLogP", "RDKit MolLogP", "B  Computed lipophilicity"),
        ("TPSA", r"TPSA ($\mathrm{\AA}^{2}$)", "C  Polar surface area"),
    )
    max_count = 0.0
    artists = []
    for axis, (descriptor, label, title) in zip(axes, definitions):
        artist = axis.hexbin(
            descriptors[descriptor], descriptors.rt,
            gridsize=46, mincnt=1, bins="log", cmap="viridis", linewidths=0,
        )
        artists.append(artist)
        max_count = max(max_count, float(np.max(artist.get_array())))
        axis.set_xlabel(label)
        axis.set_title(title, loc="left")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.12, linewidth=0.4)
    # Use one color scale for all three panels; color denotes observed row counts.
    for artist in artists:
        artist.set_clim(vmin=1, vmax=max_count)
    colorbar = fig.colorbar(artists[-1], ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Molecule count per hexagon (log scale)")
    axes[0].set_ylabel("Measured retention time (s)")
    fig.suptitle(f"Chemical descriptors and retention time in the random-split training partition (n = {len(descriptors):,})",
                 fontsize=12)
    png = figure_dir / "training_descriptor_hexbin.png"
    pdf = figure_dir / "training_descriptor_hexbin.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Training-only chemical descriptor analysis"})
    plt.close(fig)
    return png, pdf


def run(project_root: Path) -> dict:
    project_root = project_root.resolve()
    table_dir = project_root / "outputs" / "tables"
    figure_dir = project_root / "outputs" / "figures"
    log_dir = project_root / "outputs" / "logs"
    for directory in (table_dir, figure_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "chemical_analysis.log", mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Record only the three source files this supplementary analysis reads.
    input_paths = (
        project_root / "processed" / "molecules.csv",
        project_root / "splits" / "random.csv",
        project_root / "processed" / "excluded_records.csv",
    )
    input_hashes = {path.relative_to(project_root).as_posix(): sha256(path) for path in input_paths}
    training, retained = read_training_records(project_root)
    descriptors = compute_descriptors(training)
    correlation_table = correlations(descriptors)
    exclusion_table = exclusion_summary(project_root, retained)
    descriptor_path = table_dir / "training_descriptors.csv"
    correlation_path = table_dir / "training_descriptor_correlations.csv"
    exclusion_path = table_dir / "exclusion_summary.csv"
    descriptors.to_csv(descriptor_path, index=False, lineterminator="\n")
    correlation_table.to_csv(correlation_path, index=False, lineterminator="\n")
    exclusion_table.to_csv(exclusion_path, index=False, lineterminator="\n")
    png, pdf = draw_figure(descriptors, figure_dir)
    if input_hashes != {
        path.relative_to(project_root).as_posix(): sha256(path) for path in input_paths
    }:
        raise RuntimeError("An input changed during this supplementary analysis")
    summary = {
        "status": "complete",
        "purpose": "supplementary_descriptive_analysis_only",
        "descriptor_scope": "random split training partition only",
        "n_training": len(training),
        "n_training_rt_gt_300s": int(np.count_nonzero(descriptors.rt.to_numpy() > 300)),
        "descriptor_definitions": DESCRIPTOR_DEFINITIONS,
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "requested_blas_threads": 4,
        },
        "interpretation_notes": [
            "Spearman coefficients describe monotonic marginal associations, not causal effects.",
            "No hypothesis-test p-values or multiplicity-adjusted significance claims are reported.",
            "The seven descriptors are correlated with one another; these are not independent effects.",
            "The RT > 300 s subgroup is an exploratory descriptive stratum, not a verified physical dead-time cutoff.",
            "Computed MolLogP is not measured lipophilicity or pH-specific logD; molecular weight and TPSA alone do not determine retention.",
            "The full supplied canonical molecule is used without salt stripping or further standardization.",
            "No descriptor was added to model inputs and no model, split, label, or retained/excluded decision was changed.",
            "Exclusion summary compares all retained rows with parse failures using existing RT labels; this is a quality-control description, not model evaluation or tuning.",
            "A difference or similarity between excluded and retained RT distributions does not establish whether exclusion is unbiased.",
        ],
        "input_sha256": input_hashes,
        "outputs": {
            path.relative_to(project_root).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (descriptor_path, correlation_path, exclusion_path, png, pdf)
        },
    }
    (table_dir / "chemical_analysis_notes.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOG.info("Complete: %d descriptor rows, %d correlation rows, two exclusion cohorts",
             len(descriptors), len(correlation_table))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
