"""App Streamlit único: dashboard + análise de novo evento + chat + documentos.

Roda com: streamlit run src/app/streamlit_app.py
"""

import json
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.labels import CATEGORY_TO_DOC, load_doc_overrides
from src.llm.chat import analyze_event, ask_about_category
from src.rag.ingest import DOCS_DIR, VECTORSTORE_DIR, add_document, build_vectorstore
from src.similarity.build import DB_PATH, build_index

st.set_page_config(page_title="Manutenção Prescritiva", page_icon="🏭", layout="wide")

# Evento de exemplo do próprio edital (página 3) — id 114387 do banner.csv,
# categoria cocked_rotor (documentada).
EXEMPLO_EDITAL = {
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


@st.cache_resource(show_spinner="Construindo índice de similaridade (histórico de eventos)...")
def garantir_indice_similaridade() -> bool:
    if not DB_PATH.exists():
        build_index()
    return True


@st.cache_resource(show_spinner="Construindo índice de documentos (pode levar 1-2 min na primeira vez, por causa do OCR)...")
def garantir_indice_documentos() -> bool:
    if not VECTORSTORE_DIR.exists():
        build_vectorstore()
    return True


def registrar_documento(categoria: str, arquivo) -> None:
    destino = DOCS_DIR / arquivo.name
    destino.write_bytes(arquivo.getvalue())
    add_document(destino, category=categoria)
    build_index()  # atualiza has_documentation para os eventos já carregados
    st.cache_resource.clear()


garantir_indice_similaridade()
garantir_indice_documentos()

st.title("🏭 Manutenção Prescritiva")

aba = st.sidebar.radio("Navegação", ["Dashboard", "Novo evento", "Documentos"])

if aba == "Dashboard":
    st.header("Visão geral do histórico")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT category, is_problem, has_documentation, created_at FROM events",
            conn,
            parse_dates=["created_at"],
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de eventos", f"{len(df):,}")
    col2.metric("Eventos-problema", f"{int(df['is_problem'].sum()):,}")
    sem_doc = int((df["is_problem"] & (df["has_documentation"] == False)).sum())  # noqa: E712
    col3.metric("Sem documentação", f"{sem_doc:,}")

    resumo = df.groupby("category").size().reset_index(name="registros").sort_values("registros")
    fig = px.bar(
        resumo, x="registros", y="category", orientation="h",
        template="plotly_white", title="Registros por categoria",
    )
    st.plotly_chart(fig, use_container_width=True)

    diario = (
        df.assign(dia=df["created_at"].dt.date, tipo=df["is_problem"].map({True: "Problema", False: "Estado operacional"}))
        .groupby(["dia", "tipo"]).size().reset_index(name="registros")
    )
    fig2 = px.area(
        diario, x="dia", y="registros", color="tipo",
        color_discrete_map={"Estado operacional": "#2563eb", "Problema": "#f97316"},
        template="plotly_white", title="Volume diário de eventos",
    )
    st.plotly_chart(fig2, use_container_width=True)

elif aba == "Novo evento":
    st.header("Novo evento")

    if "evento_json" not in st.session_state:
        st.session_state["evento_json"] = json.dumps(EXEMPLO_EDITAL, indent=2, ensure_ascii=False)

    if st.button("Carregar exemplo do edital"):
        st.session_state["evento_json"] = json.dumps(EXEMPLO_EDITAL, indent=2, ensure_ascii=False)

    texto = st.text_area("JSON do evento", key="evento_json", height=280)

    if st.button("Analisar evento", type="primary"):
        try:
            evento = json.loads(texto)
        except json.JSONDecodeError as erro:
            st.error(f"JSON inválido: {erro}")
        else:
            try:
                with st.spinner("Buscando eventos semelhantes e consultando a documentação..."):
                    st.session_state["resultado"] = analyze_event(evento)
                st.session_state.pop("historico_chat", None)
            except Exception as erro:  # ex.: GROQ_API_KEY ausente e Ollama indisponível
                st.error(f"Erro ao consultar o LLM: {erro}")

    resultado = st.session_state.get("resultado")
    if resultado:
        tipo = resultado["tipo"]

        if tipo == "sem_problema":
            st.success(resultado["mensagem"])

        elif tipo == "sem_documentacao":
            st.warning(resultado["mensagem"])
            with st.expander("Registrar novo documento para esta categoria"):
                arquivo = st.file_uploader("PDF do procedimento", type="pdf", key="upload_sem_doc")
                if arquivo and st.button("Cadastrar documento", key="botao_upload_sem_doc"):
                    registrar_documento(resultado["categoria"], arquivo)
                    st.success("Documento cadastrado — analise o evento novamente para ver o diagnóstico.")

        elif tipo == "diagnostico":
            sim = resultado["similaridade"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Categoria", sim["categoria_provavel"])
            c2.metric("Ocorrências semelhantes", sim["quantidade_vizinhos_mesma_categoria"])
            c3.metric("Total histórico da categoria", sim["total_historico_categoria"])
            c4.metric("Frequência (eventos/dia)", f"{sim['frequencia_por_dia']:.2f}")

            st.markdown("### Instruções de solução")
            st.write(resultado["resposta"])
            st.caption(f"Fontes: {', '.join(resultado['fontes'])}")

            st.markdown("### Chat de acompanhamento")
            for autor, mensagem in st.session_state.get("historico_chat", []):
                with st.chat_message(autor):
                    st.write(mensagem)

            pergunta = st.chat_input("Pergunte algo sobre este defeito...")
            if pergunta:
                historico = st.session_state.setdefault("historico_chat", [])
                historico.append(("user", pergunta))
                try:
                    resposta = ask_about_category(pergunta, sim["categoria_provavel"])
                except Exception as erro:
                    resposta = f"Erro ao consultar o LLM: {erro}"
                historico.append(("assistant", resposta))
                st.rerun()

elif aba == "Documentos":
    st.header("Documentos cadastrados")

    resolvido = {**CATEGORY_TO_DOC, **load_doc_overrides()}
    tabela = pd.DataFrame(
        [{"categoria": c, "documento": d or "—", "documentado": d is not None} for c, d in resolvido.items()]
    ).sort_values("categoria")
    st.dataframe(tabela, use_container_width=True, hide_index=True)

    st.subheader("Registrar novo documento")
    categoria = st.selectbox("Categoria", sorted(resolvido.keys()))
    arquivo = st.file_uploader("Arquivo PDF", type="pdf", key="upload_doc_geral")
    if arquivo and st.button("Cadastrar"):
        registrar_documento(categoria, arquivo)
        st.success(f"Documento '{arquivo.name}' cadastrado para '{categoria}'. Recarregue a página para ver a mudança.")
