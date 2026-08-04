import pandas as pd
import pytest

from src.data.labels import CATEGORY_TO_DOC, build_label_mapping, normalize_fault


@pytest.mark.parametrize(
    "raw_label",
    [
        "normal",
        "normal_2",
        "normal_carga_3_3",
        "normla_carga_3_3",  # typo
        "new_baseline",
        "teste",
        "new_teste",
        "new_tes",  # truncado
        "acelerando",
        "motor_desligado",
        "mortor_desligado_novo",  # typo
    ],
)
def test_estados_nao_sao_problema(raw_label):
    result = normalize_fault(raw_label)
    assert result["is_problem"] is False
    assert result["category"] == "estado_operacional"


@pytest.mark.parametrize(
    "raw_label,categoria_esperada",
    [
        ("rolamento_inner", "rolamento"),
        ("rolamento_outer_carga_2", "rolamento"),
        ("desalinhado_2", "desalinhamento"),
        ("new_desalinhado_4", "desalinhamento"),
        ("desbalanceado_1parafuso", "desbalanceamento"),
        ("desbanlanceado_carga_3_2", "desbalanceamento"),  # typo
        ("dedesbalanceado_adxl_1", "desbalanceamento"),  # typo
        ("new_desabanceado_1", "desbalanceamento"),  # typo
        ("correia_2", "correia"),
        ("polia_2", "polia"),
        ("cocked_rotor_2", "cocked_rotor"),
        ("eccentric_rotor_carga_2", "eccentric_rotor"),
        ("ventoinha_adxl_0", "ventoinha"),
        ("new_falta_fase_1", "falta_fase"),
    ],
)
def test_categorias_de_defeito(raw_label, categoria_esperada):
    result = normalize_fault(raw_label)
    assert result["is_problem"] is True
    assert result["category"] == categoria_esperada


def test_categorias_sem_documento_sao_marcadas():
    for categoria in ("eccentric_rotor", "ventoinha", "falta_fase"):
        assert CATEGORY_TO_DOC[categoria] is None


def test_categorias_com_documento_apontam_para_arquivo_existente():
    documentados = {c: doc for c, doc in CATEGORY_TO_DOC.items() if doc}
    assert documentados == {
        "rolamento": "Doc1.pdf",
        "desalinhamento": "Doc2.pdf",
        "desbalanceamento": "Doc3.pdf",
        "correia": "Doc4.pdf",
        "polia": "Doc5.pdf",
        "cocked_rotor": "Doc6.pdf",
    }


def test_todos_os_151_rotulos_reais_sao_classificados():
    df = pd.read_csv("data/raw/banner.csv")
    mapping = build_label_mapping(df["fault"].unique())

    assert len(mapping) == 151
    assert (mapping["category"] != "desconhecido").all()
