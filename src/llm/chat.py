"""Orquestração do assistente: similaridade -> decisão -> (RAG + LLM) ou recusa.

O ponto central deste módulo é `analyze_event`: ele decide se o LLM pode ser
chamado ANTES de chamá-lo. Essa decisão é determinística (baseada só no
resultado da busca por similaridade) e não depende do LLM "se comportar bem"
— um defeito sem documentação nunca chega a virar uma chamada de LLM, então
não existe alucinação possível nesse caminho.
"""

import os

from dotenv import load_dotenv

from src.llm.prompts import SYSTEM_PROMPT, build_chat_prompt, build_diagnosis_prompt
from src.rag.ingest import retrieve_for_category
from src.similarity.search import find_similar_events

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
RAG_K = 5

_llm = None


def get_llm():
    """Cadeia de LLM com fallback: Groq (nuvem) -> Ollama (local, offline).

    Import tardio das integrações porque elas fazem chamadas de rede/checagem
    de credenciais na importação — módulos que só usam a lógica de decisão
    (ex.: testes do gate) não precisam pagar esse custo nem ter Ollama instalado.
    """
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        from langchain_ollama import ChatOllama

        primario = ChatGroq(model=GROQ_MODEL, temperature=0.2)
        fallback = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
        _llm = primario.with_fallbacks([fallback])
    return _llm


def analyze_event(event: dict, k: int = 50) -> dict:
    """Decide o que responder para um novo evento — só chama o LLM se houver
    documentação para a categoria identificada por similaridade."""
    resultado = find_similar_events(event, k=k)

    if not resultado["is_problem"]:
        return {
            "tipo": "sem_problema",
            "mensagem": resultado["mensagem"],
            "similaridade": resultado,
        }

    categoria = resultado["categoria_provavel"]

    if not resultado["has_documentation"]:
        return {
            "tipo": "sem_documentacao",
            "categoria": categoria,
            "mensagem": (
                f"Identifiquei um padrão semelhante a eventos históricos de "
                f"'{categoria}' ({resultado['quantidade_vizinhos_mesma_categoria']} "
                f"ocorrências parecidas), mas ainda não existe documentação "
                f"cadastrada para esse tipo de defeito. Registre um novo "
                f"documento de procedimento para que eu possa orientar a correção."
            ),
            "similaridade": resultado,
        }

    trechos = retrieve_for_category(
        f"diagnóstico e correção de {categoria}", category=categoria, k=RAG_K
    )
    contexto = "\n\n---\n\n".join(t.page_content for t in trechos)
    prompt = build_diagnosis_prompt(event, resultado, contexto)

    resposta = get_llm().invoke(
        [("system", SYSTEM_PROMPT), ("human", prompt)]
    )

    return {
        "tipo": "diagnostico",
        "categoria": categoria,
        "resposta": resposta.content,
        "fontes": sorted({t.metadata["doc_file"] for t in trechos}),
        "similaridade": resultado,
    }


def ask_about_category(pergunta: str, categoria: str) -> str:
    """Chat de acompanhamento: responde uma pergunta livre sobre uma categoria
    já identificada, sempre restrito ao documento daquela categoria."""
    trechos = retrieve_for_category(pergunta, category=categoria, k=RAG_K)
    if not trechos:
        return (
            f"Não encontrei conteúdo sobre isso no documento de '{categoria}'. "
            "Se achar que essa informação deveria existir, registre um "
            "documento atualizado para essa categoria."
        )

    contexto = "\n\n---\n\n".join(t.page_content for t in trechos)
    prompt = build_chat_prompt(pergunta, categoria, contexto)
    resposta = get_llm().invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
    return resposta.content
