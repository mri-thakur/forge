"""Named dataset importers with provenance and official split preservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.training import sha256

WIKITEXT_REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"


def _parquet_text_to_bytes(paths, output):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("install Forge's data extra: pip install -e '.[data]'") from error
    rows = 0
    with Path(output).open("wb") as destination:
        for path in paths:
            source = parquet.ParquetFile(path)
            if "text" not in source.schema.names:
                raise ValueError(f"{path} has no text column")
            for batch in source.iter_batches(columns=["text"], batch_size=8192):
                for value in batch.column(0).to_pylist():
                    if value:
                        destination.write(value.encode("utf-8"))
                        destination.write(b"\n")
                    rows += 1
    return rows


def prepare_wikitext(train_paths, validation_path, out):
    train_paths = [Path(path) for path in train_paths]
    validation_path, out = Path(validation_path), Path(out)
    sources = train_paths + [validation_path]
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("one or more WikiText parquet shards are missing")
    out.mkdir(parents=True, exist_ok=True)
    train_rows = _parquet_text_to_bytes(train_paths, out / "train.bin")
    val_rows = _parquet_text_to_bytes([validation_path], out / "val.bin")
    manifest = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-v1",
        "revision": WIKITEXT_REVISION,
        "homepage": "https://huggingface.co/datasets/Salesforce/wikitext",
        "licenses": ["CC-BY-SA-3.0", "GFDL"],
        "tokenizer": "utf8-bytes-256",
        "synthetic": False,
        "split": "official train/validation; nonempty rows joined with newline",
        "source_files": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sources
        ],
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_tokens": (out / "train.bin").stat().st_size,
        "val_tokens": (out / "val.bin").stat().st_size,
        "train_sha256": sha256(out / "train.bin"),
        "val_sha256": sha256(out / "val.bin"),
    }
    canonical = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
