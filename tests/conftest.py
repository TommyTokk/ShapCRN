import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def model_path() -> Path:
    return Path(__file__).parent / "data" / "minimal.xml"
