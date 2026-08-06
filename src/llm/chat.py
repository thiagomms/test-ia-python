"""Orquestração do assistente: similaridade -> decisão -> (RAG + LLM) ou recusa.

O ponto central deste módulo é `analyze_event`: ele decide se o LLM pode ser
chamado ANTES de chamá-lo. Essa decisão é determinística (baseada só no
resultado da busca por similaridade) e não depende do LLM "se comportar bem"
— um defeito sem documentação nunca chega a virar uma chamada de LLM, então
não existe alucinação possível nesse caminho.
"""

import os
import re

from dotenv import load_dotenv

from src.llm.prompts import (
    CHAT_SYSTEM_PROMPT,
    SUGESTOES_PERGUNTAS,
    SYSTEM_PROMPT,
    build_chat_prompt,
    build_diagnosis_prompt,
)
from src.rag.ingest import retrieve_for_category
from src.similarity.search import find_similar_events

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
RAG_K = 5

# Cumprimentos / mensagens sociais curtas que não devem disparar o RAG/LLM técnico.
_CUMPRIMENTOS = frozenset(
    {
        "oi",
        "olá",
        "ola",
        "oie",
        "opa",
        "eai",
        "e aí",
        "e ai",
        "hey",
        "hi",
        "hello",
        "bom dia",
        "boa tarde",
        "boa noite",
        "tudo bem",
        "tudo bom",
        "obrigado",
        "obrigada",
        "valeu",
        "thanks",
        "ok",
        "okay",
        "blz",
        "beleza",
    }
)

_llm = None


def normalizar_texto(texto: str) -> str:
    """Normaliza a mensagem para matching de intenções (minúsculas, typos comuns)."""
    t = (texto or "").strip().lower()
    t = t.replace("oque", "o que").replace("q é", "o que é").replace("q e", "o que é")
    t = re.sub(r"[^\w\sàáâãéêíóôõúç]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def eh_mensagem_social(pergunta: str) -> bool:
    """Retorna True se a mensagem for só cumprimento/agradecimento, sem pergunta técnica."""
    texto = normalizar_texto(pergunta)
    return bool(texto) and texto in _CUMPRIMENTOS


def reescrever_pergunta_para_rag(pergunta: str, categoria: str) -> str:
    """Expande perguntas informais em query alinhada às seções do procedimento."""
    texto = normalizar_texto(pergunta)

    if any(p in texto for p in ("ferramenta", "epi", "equipamento necess")):
        return (
            f"ferramentas necessárias EPI equipamentos para diagnóstico "
            f"e correção de {categoria}"
        )
    if any(p in texto for p in ("causa", "por que", "porque", "motivo")):
        return (
            f"principais causas montagem desgaste eixo cubo danos mecânicos "
            f"de {categoria}"
        )
    if any(
        p in texto
        for p in ("correç", "corrig", "passo", "como fazer", "procedimento", "reparo")
    ):
        return f"correção da falha procedimento passos de reparo de {categoria}"
    if any(
        p in texto
        for p in ("sintoma", "vibra", "ruído", "ruido", "aquece", "sinal", "barulho")
    ):
        return (
            f"sintomas comuns vibração ruído aquecimento caracterização "
            f"da falha de {categoria}"
        )

    # Perguntas vagas do tipo "o que tem / o que acontece / o que não funciona".
    vagos = (
        "o que tem",
        "que tem",
        "o que acontece",
        "o que ocorre",
        "o que nao funciona",
        "o que não funciona",
        "nao funciona",
        "não funciona",
        "o que e",
        "o que é",
        "explique",
        "me explica",
        "me fala",
        "sobre o",
        "sobre a",
        "como funciona",
        "qual o problema",
        "qual problema",
        "o problema",
        "afins",
        "em geral",
        "resumo",
        "o que falha",
        "que falha",
    )
    if any(p in texto for p in vagos):
        return (
            f"caracterização da falha introdução o que é o defeito sintomas comuns "
            f"o que acontece o que falha na operação principais causas de {categoria}"
        )

    return f"{pergunta.strip()} {categoria}"


def mensagem_sem_trechos(categoria: str) -> str:
    """Monta a resposta quando o RAG não encontra trechos úteis no documento."""
    return (
        f"Não encontrei conteúdo sobre isso no documento de '{categoria}'. "
        "Se achar que essa informação deveria existir, registre um "
        f"documento atualizado para essa categoria.\n\n{SUGESTOES_PERGUNTAS}"
    )


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
    já identificada, sempre restrito ao documento daquela categoria.

    Cumprimentos curtos (ex.: "oi") recebem resposta social sem chamar RAG/LLM.
    Perguntas informais são reescritas para melhorar a recuperação no FAISS.
    """
    if eh_mensagem_social(pergunta):
        return (
            f"Olá! Posso ajudar com dúvidas sobre o defeito '{categoria}' — "
            "por exemplo o que é o defeito, sintomas, causas, ferramentas "
            "ou passos de correção."
        )

    query_rag = reescrever_pergunta_para_rag(pergunta, categoria)
    trechos = retrieve_for_category(query_rag, category=categoria, k=RAG_K)
    if not trechos:
        return mensagem_sem_trechos(categoria)

    contexto = "\n\n---\n\n".join(t.page_content for t in trechos)
    prompt = build_chat_prompt(pergunta, categoria, contexto)
    resposta = get_llm().invoke([("system", CHAT_SYSTEM_PROMPT), ("human", prompt)])
    return resposta.content
