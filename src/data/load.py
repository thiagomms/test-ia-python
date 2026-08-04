"""Carregamento do banner.csv com enriquecimento de rótulos de fault."""

from pathlib import Path

import pandas as pd

from src.data.labels import enrich_faults

DEFAULT_BANNER_PATH = Path("data/raw/banner.csv")


def load_banner(path: Path = DEFAULT_BANNER_PATH) -> pd.DataFrame:
    """Carrega o banner.csv, parseando datas e classificando a coluna fault."""
    df = pd.read_csv(path, parse_dates=["created_at"])
    return enrich_faults(df)
