"""Busca por similaridade: dado um novo evento, encontra ocorrências históricas
com padrão de sensor semelhante e resume o que é conhecido sobre elas.

Importante: o campo `fault` que eventualmente acompanha o evento de entrada
NÃO é usado aqui — o edital pede identificação por similaridade de padrão, não
por confiança em uma classificação prévia. O rótulo devolvido (`categoria_provavel`)
vem da maioria entre os vizinhos mais próximos no espaço de features.
"""

import sqlite3

import faiss
import joblib
import pandas as pd

from src.similarity.build import DB_PATH, FAISS_PATH, SCALER_PATH
from src.similarity.features import extract_feature_vector

DEFAULT_K = 50

# O sensor é lido em um pequeno conjunto de rotações controladas (0/500/1000/
# 2000/3000 RPM), e vibração escala com a rotação (ver data/docs/Doc3.pdf) — na
# prática o rpm padronizado já domina tanto a distância euclidiana que os
# vizinhos mais próximos praticamente nunca cruzam faixas de rotação (medido:
# ver docs/DECISIONS.md). Este filtro existe como salvaguarda explícita e
# barata para esse comportamento, não como a correção principal: o limite de
# acurácia real observado vem de categorias se sobreporem DENTRO da mesma
# rotação, sobretudo em 500 RPM (falha incipiente com sinal fraco).
CANDIDATE_POOL_MULTIPLIER = 10
MIN_CANDIDATES_APOS_FILTRO = 5


def _load_scaler():
    return joblib.load(SCALER_PATH)


def _load_faiss_index():
    return faiss.read_index(str(FAISS_PATH))


def _connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def find_similar_events(event: dict, k: int = DEFAULT_K) -> dict:
    """Retorna um resumo prescritivo baseado nos k vizinhos mais próximos."""
    vector = extract_feature_vector(event).reshape(1, -1)
    scaled = _load_scaler().transform(vector).astype("float32")

    index = _load_faiss_index()
    pool_size = k * CANDIDATE_POOL_MULTIPLIER
    distances, ids = index.search(scaled, pool_size)
    pares = [(int(i), float(d)) for i, d in zip(ids[0], distances[0]) if i != -1]

    with _connection() as conn:
        placeholders = ",".join("?" * len(pares))
        candidatos = pd.read_sql(
            f"SELECT * FROM events WHERE id IN ({placeholders})",  # noqa: S608 (ids validados, não input livre)
            conn,
            params=[i for i, _ in pares],
        )

    distancia_por_id = dict(pares)
    candidatos["distance"] = candidatos["id"].map(distancia_por_id)

    mesma_rotacao = candidatos[candidatos["rpm"] == event.get("rpm")]
    pool_final = mesma_rotacao if len(mesma_rotacao) >= MIN_CANDIDATES_APOS_FILTRO else candidatos

    vizinhos = pool_final.sort_values("distance").head(k)

    categoria_provavel = vizinhos["category"].value_counts().idxmax()
    correspondentes = vizinhos[vizinhos["category"] == categoria_provavel]

    if categoria_provavel == "estado_operacional":
        return {
            "categoria_provavel": categoria_provavel,
            "is_problem": False,
            "mensagem": (
                "Os eventos históricos mais parecidos com este evento são de "
                "operação normal — não há indício de defeito."
            ),
            "total_k_vizinhos": len(vizinhos),
            "quantidade_vizinhos_mesma_categoria": len(correspondentes),
        }

    with _connection() as conn:
        total_historico = pd.read_sql(
            "SELECT COUNT(*) AS total FROM events WHERE category = ?",
            conn,
            params=[categoria_provavel],
        )["total"].iloc[0]

    periodo = pd.to_datetime(correspondentes["created_at"])
    span_days = max((periodo.max() - periodo.min()).days, 1)
    has_documentation = bool(correspondentes["has_documentation"].iloc[0])

    return {
        "categoria_provavel": categoria_provavel,
        "is_problem": True,
        "has_documentation": has_documentation,
        "doc_file": correspondentes["doc_file"].iloc[0] if has_documentation else None,
        "total_k_vizinhos": len(vizinhos),
        "quantidade_vizinhos_mesma_categoria": len(correspondentes),
        "total_historico_categoria": int(total_historico),
        "frequencia_por_dia": len(correspondentes) / span_days,
        "primeiro_evento": periodo.min().isoformat(),
        "ultimo_evento": periodo.max().isoformat(),
        "rpm_medio": float(correspondentes["rpm"].mean()),
        "distancia_media": float(correspondentes["distance"].mean()),
    }
