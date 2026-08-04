"""Seleção de features numéricas usadas na busca por similaridade.

O banner.csv reporta a mesma medição em duas unidades (ex.: `z_rms_velocity_in_s`
e `z_rms_velocity_mm_s` são a mesma grandeza convertida) e também `temperature_f`
e `temperature_c`. Manter os dois pares dobraria o peso dessas medições na
distância euclidiana. Por isso a lista abaixo mantém apenas a unidade métrica de
cada par.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "z_rms_velocity_mm_s",
    "x_rms_velocity_mm_s",
    "temperature_c",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
    "rpm",
]


def extract_feature_vector(event: dict) -> np.ndarray:
    """Converte um evento (JSON de entrada) no vetor de features esperado.

    Levanta ValueError se algum campo obrigatório estiver ausente — validação
    na borda do sistema, já que o evento vem de uma fonte externa.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in event]
    if missing:
        raise ValueError(f"Evento incompleto, faltando campos: {missing}")
    return np.array([float(event[c]) for c in FEATURE_COLUMNS], dtype="float32")


def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extrai a matriz de features de um DataFrame de eventos históricos."""
    return df[FEATURE_COLUMNS].to_numpy(dtype="float32")
