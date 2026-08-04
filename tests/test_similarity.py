import pytest

from src.data.load import load_banner
from src.similarity.build import build_index
from src.similarity.features import extract_feature_vector
from src.similarity.search import find_similar_events

# Evento de exemplo do próprio edital (página 3) — corresponde ao id 114387
# do banner.csv, categoria cocked_rotor (documentada em Doc6.pdf).
EVENTO_EXEMPLO_EDITAL = {
    "id": 114387,
    "created_at": "2026-06-01 21:32:53.911176+00:00",
    "z_rms_velocity_in_s": 0.0597,
    "z_rms_velocity_mm_s": 1.517,
    "temperature_f": 76.44,
    "temperature_c": 24.69,
    "x_rms_velocity_in_s": 0.0787,
    "x_rms_velocity_mm_s": 2.0,
    "z_peak_acceleration_g": 0.484,
    "x_peak_acceleration_g": 0.631,
    "z_peak_vel_comp_freq_hz": 61.0,
    "x_peak_vel_comp_freq_hz": 61.0,
    "z_rms_acceleration_g": 0.09,
    "x_rms_acceleration_g": 0.114,
    "z_kurtosis": 2.392,
    "x_kurtosis": 2.77,
    "z_crest_factor": 3.747,
    "x_crest_factor": 4.269,
    "z_peak_velocity_in_s": 0.0844,
    "z_peak_velocity_mm_s": 2.146,
    "x_peak_velocity_in_s": 0.1113,
    "x_peak_velocity_mm_s": 2.829,
    "z_high_freq_rms_accel_g": 0.129,
    "x_high_freq_rms_accel_g": 0.147,
    "fault": "cocked_rotor_2",
    "rpm": 1000.0,
}


@pytest.fixture(scope="module")
def df():
    return load_banner()


@pytest.fixture(scope="module", autouse=True)
def indexed(df):
    build_index(df)


def test_evento_exemplo_do_edital_e_identificado_como_cocked_rotor():
    resultado = find_similar_events(EVENTO_EXEMPLO_EDITAL, k=25)

    assert resultado["categoria_provavel"] == "cocked_rotor"
    assert resultado["is_problem"] is True
    assert resultado["has_documentation"] is True
    assert resultado["doc_file"] == "Doc6.pdf"
    assert resultado["quantidade_vizinhos_mesma_categoria"] >= 1


def test_categoria_sem_documentacao_e_sinalizada(df):
    # Em 2000 RPM a assinatura de eccentric_rotor é bem separável (~88% de
    # acerto medido em docs/DECISIONS.md); evita depender de uma linha em
    # 500 RPM, onde falhas incipientes se confundem com o ruído de base.
    linha = df[(df["category"] == "eccentric_rotor") & (df["rpm"] == 2000.0)].iloc[0]
    evento = linha.to_dict()

    resultado = find_similar_events(evento, k=25)

    assert resultado["categoria_provavel"] == "eccentric_rotor"
    assert resultado["is_problem"] is True
    assert resultado["has_documentation"] is False
    assert resultado["doc_file"] is None


def test_acuracia_geral_da_busca_por_similaridade(df):
    """Valida a acurácia agregada (não um caso isolado) contra um piso
    conservador — ver docs/DECISIONS.md para a medição completa (73,3% geral,
    com forte dependência de RPM). Serve para detectar regressões futuras."""
    categorias = [c for c in df["category"].unique() if c != "estado_operacional"]

    acertos = 0
    total = 0
    for categoria in categorias:
        amostra = df[df["category"] == categoria].sample(
            min(5, (df["category"] == categoria).sum()), random_state=42
        )
        for _, linha in amostra.iterrows():
            resultado = find_similar_events(linha.to_dict(), k=25)
            acertos += resultado["categoria_provavel"] == categoria
            total += 1

    assert acertos / total >= 0.6


def test_evento_semelhante_a_normal_nao_e_reportado_como_problema(df):
    linha = df[df["category"] == "estado_operacional"].iloc[0]
    evento = linha.to_dict()

    resultado = find_similar_events(evento, k=25)

    assert resultado["categoria_provavel"] == "estado_operacional"
    assert resultado["is_problem"] is False


def test_evento_incompleto_levanta_erro_de_validacao():
    with pytest.raises(ValueError):
        extract_feature_vector({"rpm": 1000.0})
