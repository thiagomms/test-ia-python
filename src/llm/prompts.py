"""Templates de prompt para o assistente de manutenção prescritiva."""

SYSTEM_PROMPT = """\
Você é um assistente de manutenção prescritiva industrial. Sua função é explicar
um defeito identificado em uma máquina rotativa e orientar a correção, usando
EXCLUSIVAMENTE as informações do procedimento técnico fornecido no contexto.

Regras obrigatórias:
- Baseie-se apenas no CONTEXTO fornecido. Nunca invente ferramentas, passos,
  causas ou critérios que não estejam explicitamente no texto.
- Se o contexto não cobrir algum ponto perguntado, diga isso claramente em vez
  de complementar com conhecimento próprio.
- Seja direto e prático: liste causas prováveis, ferramentas necessárias e os
  passos de correção, na ordem do procedimento.
- Responda em português.
"""


def build_diagnosis_prompt(evento: dict, resultado_similaridade: dict, contexto: str) -> str:
    categoria = resultado_similaridade["categoria_provavel"]
    return f"""\
Um novo evento foi registrado no sistema de monitoramento. A busca por
similaridade no histórico indica que ele corresponde ao padrão de defeito
"{categoria}", com {resultado_similaridade['quantidade_vizinhos_mesma_categoria']} \
ocorrências semelhantes já registradas (de um total histórico de \
{resultado_similaridade['total_historico_categoria']} registros dessa categoria) \
e frequência aproximada de {resultado_similaridade['frequencia_por_dia']:.2f} \
ocorrências/dia nesse padrão. RPM médio dos eventos semelhantes: \
{resultado_similaridade['rpm_medio']:.0f}.

Dados do evento atual:
{evento}

CONTEXTO (trechos do procedimento técnico para "{categoria}"):
\"\"\"
{contexto}
\"\"\"

Com base apenas no contexto acima, explique o que é esse defeito e quais são
as instruções de correção recomendadas.
"""


def build_chat_prompt(pergunta: str, categoria: str, contexto: str) -> str:
    return f"""\
Pergunta do técnico sobre o defeito "{categoria}": {pergunta}

CONTEXTO (trechos do procedimento técnico para "{categoria}"):
\"\"\"
{contexto}
\"\"\"

Responda com base apenas no contexto acima.
"""
