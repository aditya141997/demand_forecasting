"""Run-specific data/results folders. Set DF_RUN=n3000 to use data/n3000 and results/n3000; unset = data/, results/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("DF_RUN", "")
DATA_DIR, RESULTS_DIR = ROOT / "data" / RUN, ROOT / "results" / RUN
