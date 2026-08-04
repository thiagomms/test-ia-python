# Decisões Técnicas

Registro vivo de decisões de arquitetura e problemas relevantes encontrados durante
o desenvolvimento. Não é o README do projeto — é o histórico de "porquês" para consulta
na entrevista.

## Stack

- **LLM**: Groq API (primário) com fallback para Ollama local (offline). Escolhido por
  liberdade de ferramentas (item 4 do edital) e por deslocar o processamento pesado de
  geração de texto para a nuvem, deixando a estação local (32GB RAM / 16GB GPU) responsável
  apenas por embeddings leves (FastEmbed) e busca vetorial (FAISS) — bem abaixo do limite de
  hardware. Trade-off: depende de internet na estação industrial; mitigado pelo fallback local.
- **RAG / similaridade**: FAISS + FastEmbed, orquestrados via LangChain.
- **Interface**: Streamlit único (dashboard + chat + input de evento), para priorizar prazo.
- **Armazenamento estruturado**: em aberto (SQLite vs PostgreSQL).

## Dados

### Rótulos de `fault` (151 valores brutos → 9 categorias de defeito + estados)

Os 151 rótulos brutos são variações de um pequeno conjunto de causas reais: sufixos de
posição de sensor (`_pos_2`), carga (`_carga`), repetição de experimento (`_2`, `_3`,
`_novo`, `_antigo`) e erros de digitação (`desbanlanceado`, `dedesbalanceado`,
`mortor_desligado`, `normla_carga_3_3`). Normalização feita em `src/data/labels.py` via
regex, com verificação de cobertura total dos 151 valores (`tests/test_labels.py`).

Descoberta relevante: **~29.600 registros (17,7% da base)** pertencem a 3 categorias de
defeito reais (`eccentric_rotor`, `ventoinha`, `falta_fase`) que **não têm nenhum
documento correspondente** em `data/docs/`. Isso é o caso de uso central do requisito
"o sistema deve reportar que ainda não existe o problema identificado" — não é um caso
sintético, é maioria detectável nos próprios dados fornecidos.

### Bug encontrado: precedência estado vs. categoria em `normalize_fault`

Ao escrever `tests/test_load.py` comparando contagens por categoria, o teste falhou:
`rolamento` apareceu com 60.729 em vez de 60.779 (diferença de 50). Causa: o rótulo
`rolamento_outer_novo_teste` (50 ocorrências) contém tanto `rolamento` quanto `teste`, e a
implementação original verificava o padrão de **estado** (`teste`) antes do padrão de
**categoria de defeito** (`rolamento`), classificando incorretamente um defeito real como
"estado operacional sem problema".

Fix: inverter a ordem — checar categorias de defeito específicas primeiro, e só cair no
padrão de estado genérico se nenhuma categoria bateu. Regra geral: um nome de defeito
explícito no rótulo é mais informativo do que uma palavra de estado genérica usada como
sufixo de variação experimental.
