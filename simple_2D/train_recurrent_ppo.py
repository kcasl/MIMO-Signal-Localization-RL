"""Forwarder: main PPO training is ADWA buildings at the project root.

Running this file from simple_2D still trains on native ADWA occupancy PNGs
(13 train buildings, 4 held-out for test). Checkpoints are written under
MIMOPathFinder/checkpoints/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_adwa_ppo import RecurrentActorCritic, evaluate, gae, main, von_mises_entropy  # noqa: F401

if __name__ == "__main__":
    main()
