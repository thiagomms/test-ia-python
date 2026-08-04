from pathlib import Path

import pytest

from src.rag import ingest


def test_load_and_chunk_marca_metadados_do_documento():
    chunks = ingest.load_and_chunk(Path("data/docs/Doc1.pdf"))

    assert len(chunks) > 0
    assert all(chunk.metadata["doc_file"] == "Doc1.pdf" for chunk in chunks)
    assert all(chunk.metadata["category"] == "rolamento" for chunk in chunks)
    assert "rolamento" in chunks[0].page_content.lower()


@pytest.fixture(scope="module")
def vectorstore_construido():
    ingest.build_vectorstore()


def test_retrieve_for_category_traz_trechos_do_documento_correto(vectorstore_construido):
    resultados = ingest.retrieve_for_category(
        "quais ferramentas são necessárias para o diagnóstico?", category="correia", k=4
    )

    assert len(resultados) > 0
    assert all(r.metadata["doc_file"] == "Doc4.pdf" for r in resultados)


def test_retrieve_for_category_nao_mistura_categorias(vectorstore_construido):
    resultados = ingest.retrieve_for_category(
        "critérios de aceitação", category="cocked_rotor", k=4
    )

    assert len(resultados) > 0
    assert all(r.metadata["category"] == "cocked_rotor" for r in resultados)


def test_add_document_ingesta_e_registra_categoria_sem_documentacao(tmp_path, monkeypatch):
    import shutil

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    shutil.copy("data/docs/Doc3.pdf", docs_dir / "Doc3.pdf")  # vectorstore base, pra não ficar vazio

    overrides_isolado = tmp_path / "category_doc_overrides.json"
    monkeypatch.setattr(ingest, "VECTORSTORE_DIR", tmp_path / "docs_faiss")
    monkeypatch.setattr("src.data.labels.OVERRIDES_PATH", overrides_isolado)

    ingest.build_vectorstore(docs_dir=docs_dir)

    novo_doc = tmp_path / "Doc_eccentric_novo.pdf"
    shutil.copy("data/docs/Doc6.pdf", novo_doc)  # reaproveita um PDF real só para testar a ingestão

    ingest.add_document(novo_doc, category="eccentric_rotor")

    from src.data.labels import load_doc_overrides

    assert load_doc_overrides() == {"eccentric_rotor": "Doc_eccentric_novo.pdf"}

    resultados = ingest.retrieve_for_category(
        "inclinação do rotor", category="eccentric_rotor", k=2
    )
    assert len(resultados) > 0
    assert all(r.metadata["doc_file"] == "Doc_eccentric_novo.pdf" for r in resultados)


def test_add_document_associa_categoria_sem_documentacao(tmp_path, monkeypatch):
    overrides_isolado = tmp_path / "category_doc_overrides.json"
    monkeypatch.setattr("src.data.labels.OVERRIDES_PATH", overrides_isolado)

    from src.data.labels import load_doc_overrides, register_document_override

    assert load_doc_overrides() == {}
    register_document_override("eccentric_rotor", "Doc7_eccentric.pdf")

    assert load_doc_overrides() == {"eccentric_rotor": "Doc7_eccentric.pdf"}
