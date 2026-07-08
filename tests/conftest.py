import os
import sys
import pytest
from pathlib import Path

# make the repo-root "examples" package importable regardless of how
# pytest is invoked (bare `pytest` does not add the cwd to sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "data: requires POLYUQ_DATA_DIR pointing at refodat dataset"
    )


@pytest.fixture(scope="session")
def data_dir():
    d = os.environ.get("POLYUQ_DATA_DIR", "")
    if not d:
        pytest.skip("POLYUQ_DATA_DIR not set — skipping data-driven tests")
    p = Path(d)
    if not p.is_dir():
        pytest.skip(f"POLYUQ_DATA_DIR={d} is not a directory")
    return p
