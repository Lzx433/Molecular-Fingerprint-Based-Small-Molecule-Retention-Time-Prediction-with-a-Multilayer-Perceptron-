"""Validate SMRT inputs, derive structural groups, and create reproducible splits.

Run from any directory::

    python prepare_data.py --project-root /path/to/metlin_smrt

The raw files are never changed. row_id is the zero-based row of processed/X.npy.
This script does not filter early-eluting molecules or average duplicate labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# The project can use the desktop's bundled Python with packages installed here.
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime"))

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


SEED = 2026
FP_SIZE = 1024
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = np.array([0.8, 0.1, 0.1], dtype=np.float64)
ACYCLIC_GROUP = "__ACYCLIC__"
MOLECULE_COLUMNS = (
    "row_id", "pubchem", "rt", "inchi", "canonical_smiles",
    "structure_group", "scaffold_group",
)
LOG = logging.getLogger("smrt.prepare")


class InputValidationError(ValueError):
    """An input would produce an ambiguous or invalid learning dataset."""


class DiagnosticContext(logging.Filter):
    """Attach the current input record to RDKit diagnostics and save warnings."""

    def __init__(self) -> None:
        super().__init__()
        self.source_row: int | str = "-"
        self.pubchem = "-"
        self.warnings: list[dict[str, Any]] = []

    def filter(self, record: logging.LogRecord) -> bool:
        record.source_row = self.source_row
        record.pubchem = self.pubchem
        if record.name == "rdkit" and record.levelno >= logging.WARNING:
            self.warnings.append({
                "source_row": self.source_row,
                "pubchem": self.pubchem,
                "level": record.levelname,
                # RDKit may prefix diagnostics with a wall-clock timestamp.
                # Strip it in the CSV so reproducible data artifacts hash equally.
                "message": re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", record.getMessage()),
            })
        return True


def configure_logging(path: Path) -> DiagnosticContext:
    context = DiagnosticContext()
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(context)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s "
        "[source_row=%(source_row)s pubchem=%(pubchem)s] %(message)s"
    ))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOG.handlers.clear()
    LOG.addHandler(file_handler)
    LOG.addHandler(console)
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False

    # Keep C++/RDKit diagnostics in the audit log, associated with each input row.
    rdBase.LogToPythonLogger()
    rdkit_logger = logging.getLogger("rdkit")
    rdkit_logger.handlers.clear()
    rdkit_logger.addHandler(file_handler)
    rdkit_logger.setLevel(logging.WARNING)
    rdkit_logger.propagate = False
    return context


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_pubchem(value: str | None, location: str) -> str:
    value = (value or "").strip()
    if not re.fullmatch(r"[0-9]+", value) or int(value) <= 0:
        raise InputValidationError(f"{location}: invalid PubChem CID {value!r}")
    return str(int(value))


def read_fingerprints(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fingerprints: dict[str, np.ndarray] = {}
    identical_duplicates = 0
    records = 0
    blank_lines = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().strip()
        if not re.search(r"\bsize\s*:\s*1024\b", header, re.IGNORECASE):
            raise InputValidationError(
                f"{path.name}: expected a description header declaring size:1024"
            )
        for line_number, line in enumerate(stream, start=2):
            line = line.rstrip("\r\n")
            if not line.strip():
                blank_lines += 1
                continue
            fields = line.split("\t")
            location = f"{path.name}:{line_number}"
            if len(fields) != 2:
                raise InputValidationError(f"{location}: expected CID<TAB>fingerprint")
            cid = normalize_pubchem(fields[0], location)
            bits = fields[1].strip()
            if len(bits) != FP_SIZE or set(bits) - {"0", "1"}:
                raise InputValidationError(
                    f"{location}: fingerprint must contain exactly {FP_SIZE} binary digits"
                )
            vector = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - np.uint8(48)
            records += 1
            if cid in fingerprints:
                if not np.array_equal(fingerprints[cid], vector):
                    raise InputValidationError(
                        f"{location}: conflicting fingerprints for PubChem CID {cid}"
                    )
                identical_duplicates += 1
            else:
                fingerprints[cid] = vector
    if not fingerprints:
        raise InputValidationError(f"{path.name}: no fingerprint records")
    audit = {
        "header": header,
        "record_count": records,
        "unique_pubchem_count": len(fingerprints),
        "identical_duplicate_id_rows": identical_duplicates,
        "blank_lines_skipped": blank_lines,
        "bit_count": FP_SIZE,
    }
    LOG.info("Read %d fingerprints (%d unique CIDs)", records, len(fingerprints))
    return fingerprints, audit


def read_molecules(
    path: Path,
    fingerprints: dict[str, np.ndarray],
    context: DiagnosticContext,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    excluded: list[dict[str, Any]] = []
    raw_count = 0
    raw_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != ["pubchem", "rt", "inchi"]:
            raise InputValidationError(
                f"{path.name}: expected semicolon-separated columns pubchem;rt;inchi, "
                f"got {reader.fieldnames!r}"
            )
        for record in reader:
            raw_count += 1
            source_row = reader.line_num
            location = f"{path.name}:{source_row}"
            if None in record or any(value is None for value in record.values()):
                raise InputValidationError(f"{location}: malformed CSV row")
            cid = normalize_pubchem(record["pubchem"], location)
            context.source_row, context.pubchem = source_row, cid
            raw_ids.add(cid)
            try:
                rt = float(record["rt"])
            except (TypeError, ValueError) as error:
                raise InputValidationError(f"{location}: invalid RT {record['rt']!r}") from error
            if not math.isfinite(rt) or rt < 0:
                raise InputValidationError(f"{location}: RT must be finite and nonnegative")
            if cid not in fingerprints:
                raise InputValidationError(f"{location}: no fingerprint for PubChem CID {cid}")
            inchi = record["inchi"].strip()
            reason = ""
            canonical_smiles = ""
            scaffold_group = ""
            try:
                if not inchi.startswith("InChI="):
                    reason = "missing_or_invalid_inchi_prefix"
                else:
                    mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=True)
                    if mol is None or mol.GetNumAtoms() == 0:
                        reason = "rdkit_inchi_parse_failed"
                    else:
                        canonical_smiles = Chem.MolToSmiles(
                            mol, canonical=True, isomericSmiles=True
                        )
                        scaffold_group = MurckoScaffold.MurckoScaffoldSmiles(
                            mol=mol, includeChirality=False
                        ) or ACYCLIC_GROUP
            except Exception as error:
                reason = f"rdkit_structure_error:{type(error).__name__}:{error}"
            if reason:
                excluded.append({
                    "source_row": source_row,
                    "pubchem": cid,
                    "rt": rt,
                    "inchi": inchi,
                    "reason": reason,
                })
                LOG.warning("Excluded molecule: %s", reason)
            else:
                rows.append({
                    "row_id": len(rows),
                    "pubchem": cid,
                    "rt": rt,
                    "inchi": inchi,
                    "canonical_smiles": canonical_smiles,
                    "structure_group": canonical_smiles,
                    "scaffold_group": scaffold_group,
                })
                vectors.append(fingerprints[cid])
            if raw_count % 5000 == 0:
                LOG.info("Parsed %d CSV rows; retained %d, excluded %d",
                         raw_count, len(rows), len(excluded))
    context.source_row, context.pubchem = "-", "-"
    if not rows:
        raise InputValidationError("No valid molecules remain after structure parsing")
    matrix = np.stack(vectors).astype(np.uint8, copy=False)
    if matrix.shape != (len(rows), FP_SIZE) or not np.isin(matrix, [0, 1]).all():
        raise InputValidationError("Internal fingerprint matrix validation failed")
    raw_audit = {
        "raw_csv_rows": raw_count,
        "raw_csv_unique_pubchem": len(raw_ids),
        "retained_rows": len(rows),
        "excluded_rows": len(excluded),
        "exclusion_reasons": dict(sorted(Counter(
            row["reason"].split(":", 1)[0] for row in excluded
        ).items())),
        "fingerprint_ids_not_in_raw_csv": len(set(fingerprints) - raw_ids),
        "missing_fingerprint_policy": "fatal_error",
        "invalid_label_policy": "fatal_error",
        "invalid_structure_policy": "exclude_and_list_in_excluded_records.csv",
    }
    LOG.info("Molecules ready: %d retained; %d excluded", len(rows), len(excluded))
    return rows, matrix, excluded, raw_audit


def summarize_rt(values: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    probabilities = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        "count": int(values.size),
        "units": "seconds",
        "mean": float(values.mean()),
        "std_population": float(values.std()),
        "quantiles": {
            f"{probability:g}": float(value)
            for probability, value in zip(probabilities, np.quantile(values, probabilities))
        },
        "rt_equal_zero": int(np.count_nonzero(values == 0)),
        "rt_le_300": int(np.count_nonzero(values <= 300)),
        "rt_le_500": int(np.count_nonzero(values <= 500)),
    }


def duplicate_audit(rows: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, Any]:
    structure_members: dict[str, list[int]] = defaultdict(list)
    pubchem_members: dict[str, list[int]] = defaultdict(list)
    fingerprint_members: dict[bytes, list[int]] = defaultdict(list)
    exact_records = Counter()
    for index, row in enumerate(rows):
        structure_members[row["structure_group"]].append(index)
        pubchem_members[row["pubchem"]].append(index)
        fingerprint_members[np.packbits(matrix[index]).tobytes()].append(index)
        exact_records[(row["pubchem"], row["rt"], row["inchi"])] += 1

    def counts(groups: dict[Any, list[int]]) -> dict[str, int]:
        repeated = [members for members in groups.values() if len(members) > 1]
        return {
            "unique_groups": len(groups),
            "groups_with_multiple_rows": len(repeated),
            "rows_in_duplicate_groups": sum(map(len, repeated)),
            "extra_rows_beyond_one_per_group": sum(len(group) - 1 for group in repeated),
            "groups_with_multiple_distinct_rt_values": sum(
                len({rows[index]["rt"] for index in members}) > 1 for members in repeated
            ),
        }

    inconsistent_cids = [
        cid for cid, members in pubchem_members.items()
        if len({rows[index]["structure_group"] for index in members}) > 1
    ]
    if inconsistent_cids:
        raise InputValidationError(
            "A CID maps to different canonical structures, so the fingerprint join is "
            f"ambiguous. First CIDs: {sorted(inconsistent_cids)[:10]}"
        )
    inconsistent_scaffolds: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        inconsistent_scaffolds[row["structure_group"]].add(row["scaffold_group"])
    if any(len(scaffolds) != 1 for scaffolds in inconsistent_scaffolds.values()):
        raise InputValidationError("The same canonical structure has multiple scaffold groups")
    return {
        "policy": "keep_all_valid_rows; do_not_average_or_deduplicate_labels",
        "structure": counts(structure_members),
        "pubchem": counts(pubchem_members),
        "identical_fingerprint": {
            **counts(fingerprint_members),
            "groups_with_multiple_distinct_structures": sum(
                len({rows[index]["structure_group"] for index in members}) > 1
                for members in fingerprint_members.values()
            ),
        },
        "exact_record_duplicates_beyond_first": sum(value - 1 for value in exact_records.values()),
        "all_zero_fingerprint_rows": int(np.count_nonzero(matrix.sum(axis=1) == 0)),
    }


def make_group_split(
    rows: list[dict[str, Any]], group_column: str, seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign whole groups without using RT labels.

    Sort groups by decreasing size after a seeded shuffle (randomizing equal-size
    ties). Assign each group to the split with the greatest remaining target row
    count. This accommodates large scaffolds before distributing singleton groups.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row[group_column]].append(row["row_id"])
    if len(groups) < len(SPLIT_NAMES):
        raise InputValidationError(f"{group_column}: need at least three distinct groups")
    group_names = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_names)
    group_names.sort(key=lambda name: -len(groups[name]))
    targets = SPLIT_FRACTIONS * len(rows)
    counts = np.zeros(3, dtype=np.int64)
    assignments = np.full(len(rows), "", dtype="<U5")
    for group_number, group_name in enumerate(group_names):
        remaining_groups = len(group_names) - group_number
        empty_splits = np.flatnonzero(counts == 0)
        if remaining_groups == len(empty_splits):
            # Preserve a nonempty holdout even for very small input fixtures or
            # a dataset dominated by just a few indivisible scaffold groups.
            split_index = int(empty_splits[np.argmax((targets - counts)[empty_splits])])
        else:
            split_index = int(np.argmax(targets - counts))
        members = groups[group_name]
        assignments[members] = SPLIT_NAMES[split_index]
        counts[split_index] += len(members)
    if np.any(counts == 0) or np.any(assignments == ""):
        raise InputValidationError(f"{group_column}: empty split or unassigned rows")

    intersections: dict[str, dict[str, int]] = {}
    for column in ("structure_group", "scaffold_group", "pubchem"):
        group_sets = {
            name: {rows[index][column] for index in np.flatnonzero(assignments == name)}
            for name in SPLIT_NAMES
        }
        intersections[column] = {
            f"{first}__{second}": len(group_sets[first] & group_sets[second])
            for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
        }
    # Scaffold overlap in the random split is allowed and explicitly reported.
    # Structure and CID overlap are prohibited in both schemes.
    required_zero = {"structure_group", "pubchem", group_column}
    for column in required_zero:
        if any(intersections[column].values()):
            raise InputValidationError(f"Leakage detected for {column} in {group_column} split")
    row_sets = {
        name: set(np.flatnonzero(assignments == name).tolist()) for name in SPLIT_NAMES
    }
    row_intersections = {
        f"{first}__{second}": len(row_sets[first] & row_sets[second])
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(row_intersections.values()) or len(set.union(*row_sets.values())) != len(rows):
        raise InputValidationError("Split row partition validation failed")
    acyclic_splits = sorted({
        str(assignments[index]) for index, row in enumerate(rows)
        if row["scaffold_group"] == ACYCLIC_GROUP
    })
    if group_column == "scaffold_group" and len(acyclic_splits) > 1:
        raise InputValidationError("Acyclic molecules were split across scaffold partitions")
    summary: dict[str, Any] = {
        "group_column": group_column,
        "seed": seed,
        "algorithm": "shuffle_equal_size_ties_then_largest_groups_first_to_greatest_remaining_target",
        "label_values_used_for_assignment": False,
        "requested_fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS.tolist())),
        "actual_rows": dict(zip(SPLIT_NAMES, counts.tolist())),
        "actual_fractions": dict(zip(SPLIT_NAMES, (counts / len(rows)).tolist())),
        "unique_groups": len(groups),
        "largest_group_rows": max(map(len, groups.values())),
        "row_id_intersections": row_intersections,
        "group_intersections": intersections,
        "required_zero_intersections": sorted(required_zero),
        "acyclic_group_splits": acyclic_splits,
        "rt_by_split": {
            name: summarize_rt([rows[index]["rt"] for index in sorted(row_sets[name])])
            for name in SPLIT_NAMES
        },
    }
    LOG.info("%s split: train=%d val=%d test=%d; required group intersections=0",
             group_column, *counts.tolist())
    return assignments, summary


def fingerprint_overlap_audit(
    rows: list[dict[str, Any]], matrix: np.ndarray,
    assignments_by_scheme: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit identical model inputs without changing the structure-based splits."""
    fingerprint_members: dict[bytes, list[int]] = defaultdict(list)
    packed = np.packbits(matrix, axis=1)
    for index in range(len(rows)):
        fingerprint_members[packed[index].tobytes()].append(index)
    repeated = {key: members for key, members in fingerprint_members.items() if len(members) > 1}
    summaries: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    for scheme, assignments in assignments_by_scheme.items():
        pair_counts = {"train__val": 0, "train__test": 0, "val__test": 0}
        cross_groups = 0
        cross_rows = 0
        cross_different_structures = 0
        for key, members in repeated.items():
            counts = Counter(str(assignments[index]) for index in members)
            if len(counts) <= 1:
                continue
            cross_groups += 1
            cross_rows += len(members)
            distinct_structures = len({rows[index]["structure_group"] for index in members})
            cross_different_structures += int(distinct_structures > 1)
            for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
                if counts[first] and counts[second]:
                    pair_counts[f"{first}__{second}"] += 1
            details.append({
                "scheme": scheme,
                "fingerprint_sha256": hashlib.sha256(key).hexdigest(),
                "unique_structures": distinct_structures,
                "total_rows": len(members),
                **{f"{name}_rows": counts[name] for name in SPLIT_NAMES},
            })
        summaries[scheme] = {
            "groups_spanning_multiple_splits": cross_groups,
            "rows_in_cross_split_groups": cross_rows,
            "cross_split_groups_with_distinct_structures": cross_different_structures,
            "fingerprint_group_intersections": pair_counts,
            "split_changed_to_remove_identical_inputs": False,
            "limitation": (
                "Structurally distinct molecules may have identical supplied 1024-bit "
                "fingerprints. This overlap is audited, not removed by the prescribed "
                "structure/scaffold splits, and limits claims about unseen model inputs."
            ),
        }
        LOG.info("%s: %d identical fingerprint groups cross splits (%d cross train/test)",
                 scheme, cross_groups, pair_counts["train__test"])
    return summaries, details


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    processed = project_root / "processed"
    splits_dir = project_root / "splits"
    processed.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    context = configure_logging(processed / "preparation.log")
    LOG.info("Preparing SMRT data under %s", project_root)
    raw_paths = {
        "dataset": project_root / "raw" / "SMRT_dataset.csv",
        "fingerprints": project_root / "raw" / "SMRT_ECFP_1024_Fingerprints.txt",
    }
    for path in raw_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required raw input not found: {path}")
    inputs = {
        key: {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in raw_paths.items()
    }
    fingerprints, fingerprint_audit = read_fingerprints(raw_paths["fingerprints"])
    rows, matrix, excluded, dataset_audit = read_molecules(
        raw_paths["dataset"], fingerprints, context
    )
    duplicates = duplicate_audit(rows, matrix)
    random_split, random_audit = make_group_split(rows, "structure_group")
    scaffold_split, scaffold_audit = make_group_split(rows, "scaffold_group")
    overlap_audit, overlap_details = fingerprint_overlap_audit(
        rows, matrix, {"random": random_split, "scaffold": scaffold_split}
    )
    scaffold_counts = Counter(row["scaffold_group"] for row in rows)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "seed": SEED,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "inputs": inputs,
        "dataset": dataset_audit,
        "fingerprints": {
            **fingerprint_audit,
            "output_shape": list(matrix.shape),
            "output_dtype": str(matrix.dtype),
            "join_key": "pubchem",
            "row_alignment": "X[row_id] matches molecules.csv row_id",
        },
        "structure_processing": {
            "parser": "RDKit MolFromInchi(sanitize=True, removeHs=True)",
            "structure_group": "canonical isomeric SMILES",
            "scaffold_group": "Murcko scaffold SMILES, includeChirality=False",
            "empty_scaffold_policy": f"one shared group: {ACYCLIC_GROUP}",
            "unique_scaffolds": len(scaffold_counts),
            "acyclic_rows": scaffold_counts.get(ACYCLIC_GROUP, 0),
            "salt_stripping": False,
            "tautomer_standardization": False,
            "rt_filter_applied": False,
            "rdkit_warning_or_error_events": len(context.warnings),
        },
        "rt": summarize_rt([row["rt"] for row in rows]),
        "duplicates": duplicates,
        "identical_fingerprint_overlap": overlap_audit,
        "splits": {"random": random_audit, "scaffold": scaffold_audit},
    }
    # Prepare all outputs before replacing any final artifact. audit.json is the
    # completion marker, published last and containing hashes of the other files.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=processed) as temp_dir:
        staging = Path(temp_dir)
        np.save(staging / "X.npy", matrix, allow_pickle=False)
        write_csv(staging / "molecules.csv", MOLECULE_COLUMNS, rows)
        write_csv(staging / "excluded_records.csv",
                  ("source_row", "pubchem", "rt", "inchi", "reason"), excluded)
        write_csv(staging / "structure_warnings.csv",
                  ("source_row", "pubchem", "level", "message"), context.warnings)
        write_csv(staging / "fingerprint_overlap.csv",
                  ("scheme", "fingerprint_sha256", "unique_structures", "total_rows",
                   "train_rows", "val_rows", "test_rows"), overlap_details)
        for name, assignments in (("random", random_split), ("scaffold", scaffold_split)):
            write_csv(staging / f"{name}.csv", ("row_id", "split"), [
                {"row_id": index, "split": str(split)}
                for index, split in enumerate(assignments)
            ])
        destinations = {
            "X.npy": processed / "X.npy",
            "molecules.csv": processed / "molecules.csv",
            "excluded_records.csv": processed / "excluded_records.csv",
            "structure_warnings.csv": processed / "structure_warnings.csv",
            "fingerprint_overlap.csv": processed / "fingerprint_overlap.csv",
            "random.csv": splits_dir / "random.csv",
            "scaffold.csv": splits_dir / "scaffold.csv",
        }
        audit["outputs"] = {
            destination.relative_to(project_root).as_posix(): {
                "sha256": sha256_file(staging / name),
                "bytes": (staging / name).stat().st_size,
            }
            for name, destination in destinations.items()
        }
        (staging / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name, destination in destinations.items():
            os.replace(staging / name, destination)
        os.replace(staging / "audit.json", processed / "audit.json")
    LOG.info("Done. Wrote processed/X.npy, molecules.csv, audit.json and both splits.")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Project directory containing raw/ (default: parent of src/)",
    )
    arguments = parser.parse_args()
    try:
        prepare(arguments.project_root)
    except Exception:
        LOG.exception("Data preparation failed; no successful completion is implied")
        raise


if __name__ == "__main__":
    main()
