"""Forwarder: evaluate the ADWA PPO checkpoint from the project root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_adwa_policy import main  # noqa: F401

if __name__ == "__main__":
    main()
