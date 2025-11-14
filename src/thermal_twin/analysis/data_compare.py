"""Data ingestion + RMSE calculations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataComparator:
    """Loads logger data and compares against simulation probes."""

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        return pd.read_excel(self.filepath)

    def rmse(self, df_sim: pd.DataFrame, df_exp: pd.DataFrame) -> float:
        sim_aligned = df_sim.set_index("time")
        exp_aligned = df_exp.set_index("time")
        diff = sim_aligned.subtract(exp_aligned)
        squared = diff.pow(2.0)
        mse = float(squared.to_numpy().mean())
        return mse ** 0.5
