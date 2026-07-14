from pathlib import Path

import pytest

import os


os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def model_path() -> Path:
    return Path(__file__).parent / "data" / "minimal.xml"
