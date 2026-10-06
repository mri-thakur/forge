import json

import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as parquet  # noqa: E402

from forge.datasets import WIKITEXT_REVISION, prepare_wikitext  # noqa: E402


def test_wikitext_import_preserves_splits_and_provenance(tmp_path):
    first, second, validation = (
        tmp_path / "a.parquet",
        tmp_path / "b.parquet",
        tmp_path / "v.parquet",
    )
    parquet.write_table(pyarrow.table({"text": ["one", "", "two"]}), first)
    parquet.write_table(pyarrow.table({"text": ["three"]}), second)
    parquet.write_table(pyarrow.table({"text": ["held out"]}), validation)
    manifest = prepare_wikitext([first, second], validation, tmp_path / "prepared")
    assert (tmp_path / "prepared" / "train.bin").read_bytes() == b"one\ntwo\nthree\n"
    assert (tmp_path / "prepared" / "val.bin").read_bytes() == b"held out\n"
    assert manifest["revision"] == WIKITEXT_REVISION
    assert manifest["licenses"] == ["CC-BY-SA-3.0", "GFDL"]
    assert json.loads((tmp_path / "prepared" / "manifest.json").read_text()) == manifest
