"""Testa a lógica de decisão (gate) de src.llm.chat sem depender de LLM real:
Groq/Ollama são substituídos por um stub, então estes testes não precisam de
GROQ_API_KEY nem de Ollama instalado — validam só a regra de negócio.
"""

from types import SimpleNamespace

import pytest

from src.llm import chat


class LLMStub:
    def __init__(self):
        self.chamadas = []

    def invoke(self, mensagens):
        self.chamadas.append(mensagens)
        return SimpleNamespace(content="resposta gerada pelo LLM")


def _resultado(**overrides):
    base = {
        "categoria_provavel": "rolamento",
        "is_problem": True,
        "has_documentation": True,
        "doc_file": "Doc1.pdf",
        "total_k_vizinhos": 50,
        "quantidade_vizinhos_mesma_categoria": 40,
        "total_historico_categoria": 60779,
        "frequencia_por_dia": 5.0,
        "primeiro_evento": "2026-05-01T00:00:00",
        "ultimo_evento": "2026-06-01T00:00:00",
        "rpm_medio": 2000.0,
        "distancia_media": 0.1,
    }
    base.update(overrides)
    return base


def test_evento_sem_problema_nao_chama_llm_nem_rag(monkeypatch):
    llm_stub = LLMStub()
    monkeypatch.setattr(chat, "find_similar_events", lambda event, k=50: {
        "categoria_provavel": "estado_operacional",
        "is_problem": False,
        "mensagem": "tudo normal",
    })
    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(chat, "retrieve_for_category", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("não deveria consultar o RAG quando não há problema")
    ))

    resultado = chat.analyze_event({"rpm": 1000.0})

    assert resultado["tipo"] == "sem_problema"
    assert llm_stub.chamadas == []


def test_categoria_sem_documentacao_nao_chama_llm_nem_rag(monkeypatch):
    """O caso mais importante: garante que o sistema nunca inventa uma
    correção para um defeito sem documento — nem chega a montar um prompt."""
    llm_stub = LLMStub()
    monkeypatch.setattr(
        chat, "find_similar_events",
        lambda event, k=50: _resultado(
            categoria_provavel="eccentric_rotor", has_documentation=False, doc_file=None
        ),
    )
    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(chat, "retrieve_for_category", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("não deveria consultar o RAG sem documentação")
    ))

    resultado = chat.analyze_event({"rpm": 500.0})

    assert resultado["tipo"] == "sem_documentacao"
    assert resultado["categoria"] == "eccentric_rotor"
    assert "registre um" in resultado["mensagem"].lower() or "registrar" in resultado["mensagem"].lower() or "novo documento" in resultado["mensagem"].lower()
    assert llm_stub.chamadas == []


def test_categoria_documentada_consulta_rag_e_chama_llm(monkeypatch):
    llm_stub = LLMStub()
    chamadas_rag = []

    def retrieve_stub(query, category, k):
        chamadas_rag.append((query, category, k))
        return [
            SimpleNamespace(page_content="trecho 1 do procedimento", metadata={"doc_file": "Doc1.pdf", "category": "rolamento"}),
            SimpleNamespace(page_content="trecho 2 do procedimento", metadata={"doc_file": "Doc1.pdf", "category": "rolamento"}),
        ]

    monkeypatch.setattr(chat, "find_similar_events", lambda event, k=50: _resultado())
    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(chat, "retrieve_for_category", retrieve_stub)

    resultado = chat.analyze_event({"rpm": 2000.0})

    assert resultado["tipo"] == "diagnostico"
    assert resultado["categoria"] == "rolamento"
    assert resultado["resposta"] == "resposta gerada pelo LLM"
    assert resultado["fontes"] == ["Doc1.pdf"]
    assert len(chamadas_rag) == 1
    assert chamadas_rag[0][1] == "rolamento"
    assert len(llm_stub.chamadas) == 1


def test_ask_about_category_sem_trechos_nao_chama_llm(monkeypatch):
    llm_stub = LLMStub()
    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(chat, "retrieve_for_category", lambda *a, **k: [])

    resposta = chat.ask_about_category("qual o torque recomendado?", "polia")

    assert "não encontrei" in resposta.lower()
    assert "sugestões" in resposta.lower()
    assert llm_stub.chamadas == []


def test_ask_about_category_cumprimento_nao_chama_rag_nem_llm(monkeypatch):
    """"oi" e similares devem gerar resposta social curta, sem procedimento técnico."""
    llm_stub = LLMStub()
    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(
        chat,
        "retrieve_for_category",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("não deveria consultar o RAG em cumprimento")
        ),
    )

    resposta = chat.ask_about_category("oi", "cocked_rotor")

    assert "olá" in resposta.lower()
    assert "cocked_rotor" in resposta
    assert llm_stub.chamadas == []


def test_reescrever_pergunta_vaga_expande_para_caracterizacao():
    """Perguntas como 'oque tem no rotor?' devem buscar sintoma/causa/caracterização."""
    query = chat.reescrever_pergunta_para_rag("oque tem no rotor?", "cocked_rotor")

    assert "cocked_rotor" in query
    assert "sintomas" in query.lower() or "caracterização" in query.lower()
    assert "causas" in query.lower()


def test_ask_about_category_usa_query_reescrita(monkeypatch):
    """Garante que o RAG recebe a query expandida, não só o texto cru do usuário."""
    llm_stub = LLMStub()
    chamadas_rag = []

    def retrieve_stub(query, category, k):
        chamadas_rag.append(query)
        return [
            SimpleNamespace(
                page_content="rotor inclinado com vibração elevada",
                metadata={"doc_file": "Doc6.pdf", "category": "cocked_rotor"},
            )
        ]

    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(chat, "retrieve_for_category", retrieve_stub)

    resposta = chat.ask_about_category("oque tem no rotor?", "cocked_rotor")

    assert resposta == "resposta gerada pelo LLM"
    assert len(chamadas_rag) == 1
    assert "sintomas" in chamadas_rag[0].lower() or "caracterização" in chamadas_rag[0].lower()


def test_ask_about_category_com_trechos_chama_llm(monkeypatch):
    llm_stub = LLMStub()
    monkeypatch.setattr(chat, "get_llm", lambda: llm_stub)
    monkeypatch.setattr(
        chat, "retrieve_for_category",
        lambda *a, **k: [SimpleNamespace(page_content="trecho", metadata={"doc_file": "Doc5.pdf", "category": "polia"})],
    )

    resposta = chat.ask_about_category("qual o torque recomendado?", "polia")

    assert resposta == "resposta gerada pelo LLM"
    assert len(llm_stub.chamadas) == 1
    # Chat de acompanhamento usa o prompt específico, não o de diagnóstico completo.
    system_msg = llm_stub.chamadas[0][0][1]
    assert "proporcional" in system_msg.lower() or "acompanhamento" in system_msg.lower()
