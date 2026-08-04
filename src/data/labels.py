"""Normalização dos rótulos brutos da coluna `fault` do banner.csv.

Os 151 rótulos brutos são variações (posição de sensor, carga, repetição de
experimento, erros de digitação) de um número pequeno de categorias reais.
Este módulo classifica cada rótulo em:

- estado operacional (não é problema: normal, baseline, teste, acelerando,
  motor_desligado) — conforme item 6 do edital;
- categoria de defeito + se existe documento de procedimento associado
  (data/docs), usado para decidir se o sistema pode prescrever uma correção
  ou deve reportar "problema sem documentação".
"""

import json
import re
from pathlib import Path

# Documentos registrados em tempo de execução pelo usuário (via app/chat) para
# categorias que nasceram sem documentação (eccentric_rotor, ventoinha,
# falta_fase). Fica em data/index/ porque é estado derivado/mutável, não
# versionado — ver .gitignore.
OVERRIDES_PATH = Path("data/index/category_doc_overrides.json")

STATE_PATTERN = re.compile(
    r"normal|normla|baseline|teste|_tes$|acelerando|desligado", re.IGNORECASE
)

# Ordem não importa: os padrões são mutuamente exclusivos entre categorias de
# defeito. Cada padrão cobre variações de posição/carga/repetição e os erros
# de digitação observados nos dados (ex.: "desbanlanceado", "dedesbalanceado").
CATEGORY_PATTERNS = [
    ("rolamento", re.compile(r"rolamento")),
    ("desalinhamento", re.compile(r"desalinhad")),
    (
        "desbalanceamento",
        re.compile(
            r"desbalance|desabalance|desbanlance|dedesbalance|ddesbalance|desabanceado"
        ),
    ),
    ("correia", re.compile(r"correia")),
    ("polia", re.compile(r"polia")),
    ("cocked_rotor", re.compile(r"cocked")),
    ("eccentric_rotor", re.compile(r"eccentric")),
    ("ventoinha", re.compile(r"ventoinha")),
    ("falta_fase", re.compile(r"falta_fase")),
]

# Categoria de defeito -> arquivo em data/docs com o procedimento de
# diagnóstico/correção. None = defeito real, porém sem documentação ainda.
CATEGORY_TO_DOC = {
    "rolamento": "Doc1.pdf",
    "desalinhamento": "Doc2.pdf",
    "desbalanceamento": "Doc3.pdf",
    "correia": "Doc4.pdf",
    "polia": "Doc5.pdf",
    "cocked_rotor": "Doc6.pdf",
    "eccentric_rotor": None,
    "ventoinha": None,
    "falta_fase": None,
}


def load_doc_overrides() -> dict:
    """Lê os documentos registrados em tempo de execução (se existir o arquivo)."""
    if OVERRIDES_PATH.exists():
        return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {}


def register_document_override(category: str, doc_file: str) -> None:
    """Associa um documento recém-cadastrado a uma categoria sem procedimento.

    Usado pelo fluxo do app quando o usuário registra um novo documento para
    um defeito sem documentação (ver Figura 01 do edital). Depois de chamar
    isto, o índice de similaridade precisa ser reconstruído
    (src.similarity.build.build_index) para que has_documentation reflita a
    mudança nos eventos já carregados.
    """
    overrides = load_doc_overrides()
    overrides[category] = doc_file
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_fault(raw_label: str, category_to_doc: dict | None = None) -> dict:
    """Classifica um rótulo bruto de `fault` em estado ou categoria de defeito."""
    label = raw_label.lower()
    category_to_doc = category_to_doc or CATEGORY_TO_DOC

    # Nome de defeito específico vence palavra de estado genérica: rótulos como
    # "rolamento_outer_novo_teste" são um defeito de rolamento repetido em uma
    # rodada de teste, não um estado "teste" sem problema.
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(label):
            doc_file = category_to_doc.get(category)
            return {
                "raw": raw_label,
                "is_problem": True,
                "category": category,
                "has_documentation": doc_file is not None,
                "doc_file": doc_file,
            }

    if STATE_PATTERN.search(label):
        return {
            "raw": raw_label,
            "is_problem": False,
            "category": "estado_operacional",
            "has_documentation": None,
            "doc_file": None,
        }

    return {
        "raw": raw_label,
        "is_problem": True,
        "category": "desconhecido",
        "has_documentation": False,
        "doc_file": None,
    }


def build_label_mapping(unique_labels) -> "pd.DataFrame":
    """Aplica normalize_fault sobre os rótulos únicos e retorna um DataFrame."""
    import pandas as pd

    category_to_doc = {**CATEGORY_TO_DOC, **load_doc_overrides()}
    mapping = pd.DataFrame(
        normalize_fault(label, category_to_doc) for label in unique_labels
    )
    # dtype nullable ("boolean", não "bool") porque estados operacionais têm
    # has_documentation = None (não se aplica) — bool comum não aceita None.
    mapping["has_documentation"] = mapping["has_documentation"].astype("boolean")
    return mapping


def enrich_faults(df: "pd.DataFrame", fault_column: str = "fault") -> "pd.DataFrame":
    """Adiciona is_problem, category, has_documentation e doc_file ao DataFrame."""
    mapping = build_label_mapping(df[fault_column].unique()).set_index("raw")
    enriched = df.join(mapping, on=fault_column)
    return enriched
