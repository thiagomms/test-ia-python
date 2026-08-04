from src.data.load import load_banner


def test_load_banner_adiciona_colunas_de_enriquecimento():
    df = load_banner()

    assert len(df) == 166796
    for column in ("is_problem", "category", "has_documentation", "doc_file"):
        assert column in df.columns

    assert df["created_at"].dtype.kind == "M"  # datetime


def test_contagem_por_categoria_bate_com_o_esperado():
    df = load_banner()
    counts = df.loc[df["is_problem"], "category"].value_counts()

    assert counts["rolamento"] == 60779
    assert counts["eccentric_rotor"] == 16497
    assert counts["ventoinha"] == 12299
    assert counts["falta_fase"] == 800

    sem_doc = df.loc[df["is_problem"] & ~df["has_documentation"]]
    assert len(sem_doc) == 16497 + 12299 + 800
