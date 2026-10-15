"""Predict SMRT-condition RT from compatible original Dragon ECFP vectors.

A matching description header checks the declared fingerprint configuration,
not the provenance of the generator. Headerless files require an explicit
compatibility declaration. Arbitrary RDKit Morgan vectors are not compatible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HEADER = (
    "Extended Connectivity FINGERPRINTS - size:1024 - length:0 to 2 - "
    "bits per pattern:2 - count fragments - atom options:Atom type|Aromaticity|"
    "Connectivity (total)|Charge|Bond order"
)


def normalized_header(value: str) -> str:
    return " ".join(value.split())


def read_fingerprints(path: Path, confirm_original_spec: bool = False):
    """Validate all records before loading a model or writing predictions."""
    records, seen = [], set()
    declaration = None
    first = True
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if "\t" not in line:
                    if normalized_header(line) != normalized_header(ORIGINAL_HEADER):
                        raise ValueError(
                            f"Line {line_number}: fingerprint description does not match "
                            "the original Dragon configuration. A 1024-bit length alone "
                            "does not establish compatibility."
                        )
                    declaration = "matching_original_description_header"
                    continue
                if not confirm_original_spec:
                    raise ValueError(
                        "Headerless input requires --confirm-original-fingerprint-spec. "
                        "Use this only after confirming the same original Dragon generator "
                        "and encoding; do not substitute RDKit Morgan fingerprints."
                    )
                declaration = "explicit_user_confirmation_for_headerless_input"
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(f"Line {line_number}: expected CID<TAB>1024 binary digits")
            cid, bits = (part.strip() for part in fields)
            if not re.fullmatch(r"[0-9]+", cid) or int(cid) <= 0:
                raise ValueError(f"Line {line_number}: PubChem CID must be a positive integer")
            cid = str(int(cid))
            if cid in seen:
                raise ValueError(f"Line {line_number}: duplicate PubChem CID {cid}")
            if len(bits) != 1024 or set(bits) - {"0", "1"}:
                raise ValueError(f"Line {line_number}: fingerprint must contain exactly 1024 bits")
            seen.add(cid)
            records.append((cid, bits.encode("ascii")))
    if not records:
        raise ValueError("Input contains no fingerprint records")
    return records, declaration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--protocol", choices=["random", "scaffold"], default="random")
    parser.add_argument("--model", choices=["mlp", "rf"], default="mlp")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", "--outputpath", type=Path, default=None,
                        help="CSV destination; default: PROJECT/outputs/new_predictions.csv")
    parser.add_argument("--confirm-original-fingerprint-spec", action="store_true",
                        help="Declare original Dragon compatibility for a headerless input")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")

    try:
        root = args.project_root.resolve()
        fingerprint_path = args.fingerprints.resolve()
        output = (args.output or root / "outputs/new_predictions.csv").resolve()
        if output.suffix.lower() != ".csv":
            raise ValueError("--output must have a .csv extension")
        if output == fingerprint_path:
            raise ValueError("Output must not overwrite the fingerprint input")
        for folder in ["raw", "processed", "splits", "src", "outputs/models"]:
            if output.is_relative_to(root / folder):
                raise ValueError(f"Do not write predictions into the protected project folder {folder}")
        records, declaration = read_fingerprints(
            fingerprint_path, args.confirm_original_fingerprint_spec
        )

        # train.py supplies the exact network and inverse target transformation,
        # and configures the project's optional local dependency directories.
        from train import Network, predict_mlp, np, torch, joblib
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        model_stem = root / f"outputs/models/{args.protocol}_{args.model}_seed{args.seed}"
        model_path = model_stem.with_suffix(".pt" if args.model == "mlp" else ".joblib")
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if args.model == "mlp":
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            metadata = checkpoint["metadata"]
            model = Network()
            model.load_state_dict(checkpoint["state_dict"])
            mu, sd = float(metadata["target_mean"]), float(metadata["target_std"])
            if not math.isfinite(mu) or not math.isfinite(sd) or sd <= 0:
                raise ValueError("The saved model has invalid target scaling")
        else:
            model = joblib.load(model_path)
            model.n_jobs = args.threads
            metadata = json.loads(model_stem.with_suffix(".json").read_text(encoding="utf-8"))
            if int(model.n_features_in_) != 1024:
                raise ValueError("Saved random forest does not expect 1024 input features")
        if metadata.get("model") != args.model or int(metadata.get("seed", -1)) != args.seed:
            raise ValueError("Saved model metadata does not match the requested model and seed")

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output.parent,
                prefix=output.stem + "_", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow(["pubchem", "predicted_rt_s", "protocol", "model", "seed"])
                for begin in range(0, len(records), 4096):
                    batch = records[begin:begin + 4096]
                    x = np.frombuffer(b"".join(bits for _, bits in batch), dtype=np.uint8)
                    x = (x.reshape(len(batch), 1024) - 48).astype(np.float32)
                    if args.model == "mlp":
                        prediction = predict_mlp(model, x, mu, sd)
                    else:
                        prediction = model.predict(x)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Model produced non-finite predictions")
                    writer.writerows(
                        (cid, format(float(value), ".10g"), args.protocol, args.model, args.seed)
                        for (cid, _), value in zip(batch, prediction)
                    )
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        provenance = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": str(fingerprint_path), "input_sha256": sha256(fingerprint_path),
            "model_file": str(model_path), "model_sha256": sha256(model_path),
            "protocol": args.protocol, "model": args.model, "seed": args.seed,
            "n_predictions": len(records), "rt_unit": "seconds", "prediction_clipping": False,
            "fingerprint_description": ORIGINAL_HEADER,
            "compatibility_basis": declaration,
            "compatibility_limit": "A declaration/header is checked; generator provenance is not independently established.",
            "scope": "Predicted absolute RT in the original SMRT chromatographic conditions; no external-system validation or uncertainty interval is implied."
        }
        output.with_suffix(".metadata.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(records):,} predictions (seconds): {output}")
        print("Use only original-compatible Dragon ECFP; predictions apply to the SMRT conditions.")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
