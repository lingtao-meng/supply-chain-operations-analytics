"""Shared data loading for the IE analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE_COL = "order date (DateOrders)"
SHIP_COL = "shipping date (DateOrders)"
SCHED_COL = "Days for shipment (scheduled)"
DATA_NAME = "DataCoSupplyChainDataset.csv"


def find_data(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        ROOT / "data" / DATA_NAME,
        Path.home() / "Desktop" / "supply-chain-ml-pipeline" / "data" / DATA_NAME,
    ]
    candidates += sorted(Path.home().glob(f".cache/kagglehub/datasets/**/{DATA_NAME}"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("DataCoSupplyChainDataset.csv not found; pass --data")


def load_orders(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="latin1",
        usecols=[DATE_COL, SHIP_COL, SCHED_COL, "Shipping Mode", "Market", "Category Name", "Order Item Quantity"],
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[SHIP_COL] = pd.to_datetime(df[SHIP_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["transit_days"] = (df[SHIP_COL] - df[DATE_COL]).dt.days
    df["delay_days"] = df["transit_days"] - df[SCHED_COL]
    return df
