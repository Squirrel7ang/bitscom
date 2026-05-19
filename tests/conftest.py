import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# PyTorch's DETAIL distributed debug mode wraps custom backends with a monitored
# ProcessGroupWrapper. The lowbit backend does not expose sequence numbers yet,
# so tests must not inherit this setting from a caller's shell environment.
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

try:
    import torch.distributed as dist

    if hasattr(dist, "DebugLevel") and hasattr(dist, "set_debug_level"):
        dist.set_debug_level(dist.DebugLevel.OFF)
except Exception:
    pass


@pytest.fixture
def run_perf_enabled():
    return os.getenv("BITSCOM_RUN_PERF", "0") == "1"
