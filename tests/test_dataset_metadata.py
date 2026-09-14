"""Testes da validação do metadata agregado do BBBC038."""

import pandas as pd
import pytest

from src.dataset import _read_metadata_image_groups


def test_read_metadata_image_groups_filters_training(monkeypatch):
    table = pd.DataFrame(
        {
            "image_set": ["training", "validation", "training", "training"],
            "image_group": ["Default", "Purple", "TissueBW", "Default"],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: table)

    assert _read_metadata_image_groups("metadata.xlsx") == ["Default", "TissueBW"]


def test_read_metadata_image_groups_rejects_missing_schema(monkeypatch):
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"image_set": ["training"]}),
    )

    with pytest.raises(ValueError, match="image_group"):
        _read_metadata_image_groups("metadata.xlsx")
