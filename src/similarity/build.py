"""Constrói os artefatos de similaridade a partir do histórico (banner.csv).

Gera três artefatos em data/index/ (não versionados — ver .gitignore — pois são
derivados e regeneráveis a partir do banner.csv):

- events.db     SQLite com todos os eventos históricos + enriquecimento de fault
                (fonte de metadados: categoria, documentação, timestamp etc.);
- events.faiss  índice FAISS (IndexFlatL2 + IDMap) das features escaladas;
- scaler.joblib StandardScaler ajustado ao histórico, usado para escalar
                novos eventos antes da busca.
"""

import sqlite3
from pathlib import Path

import faiss
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load import load_banner
from src.similarity.features import FEATURE_COLUMNS, extract_feature_matrix

INDEX_DIR = Path("data/index")
DB_PATH = INDEX_DIR / "events.db"
FAISS_PATH = INDEX_DIR / "events.faiss"
SCALER_PATH = INDEX_DIR / "scaler.joblib"


def build_index(df: pd.DataFrame | None = None) -> None:
    """Popula o SQLite de eventos e o índice FAISS a partir do histórico."""
    if df is None:
        df = load_banner()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("events", conn, if_exists="replace", index=False)

    scaler = StandardScaler()
    matrix = extract_feature_matrix(df)
    scaled = scaler.fit_transform(matrix).astype("float32")

    index = faiss.IndexIDMap(faiss.IndexFlatL2(len(FEATURE_COLUMNS)))
    index.add_with_ids(scaled, df["id"].to_numpy(dtype="int64"))

    faiss.write_index(index, str(FAISS_PATH))
    joblib.dump(scaler, SCALER_PATH)
