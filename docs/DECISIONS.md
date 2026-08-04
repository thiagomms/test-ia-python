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

## Similaridade (src/similarity)

### Features escolhidas

17 features numéricas do banner.csv, excluindo pares redundantes de unidade
(`z_rms_velocity_in_s` vs `z_rms_velocity_mm_s`, `temperature_f` vs `temperature_c` etc.)
para não dobrar o peso da mesma grandeza na distância euclidiana. `rpm` entra como feature
normal (ver abaixo por que isso já é suficiente para segregar por rotação).

### Acurácia medida da busca por similaridade (k=25, k-NN + StandardScaler + FAISS)

Amostragem de 20 eventos por categoria de defeito (semente fixa), comparando a categoria
prevista (maioria entre os 25 vizinhos mais próximos) com a categoria real:

| Categoria | Acurácia |
|---|---|
| rolamento | 95% |
| correia | 85% |
| desalinhamento | 80% |
| falta_fase | 80% |
| ventoinha | 75% |
| polia | 75% |
| eccentric_rotor | 70% |
| desbalanceamento | 65% |
| cocked_rotor | 35% |
| **Geral** | **73,3%** |

**Achado:** a acurácia depende fortemente do RPM do evento. Em 500 RPM (rotação baixa),
a acurácia despenca (ex.: `cocked_rotor` 14%, `polia` e `falta_fase` 0%); em 2000+ RPM,
sobe para 80–100% na maioria das categorias. Isso é consistente com a física descrita em
`data/docs/Doc3.pdf` ("quanto maior a rotação, maior a vibração causada pelo defeito") —
em baixa rotação, o sinal da falha ainda incipiente se aproxima do ruído de base, tornando
categorias diferentes numericamente parecidas.

Tentativa de correção: restringir os vizinhos à mesma faixa de RPM do evento antes da
votação por maioria (`src/similarity/search.py`). Resultado: **nenhuma mudança na
acurácia** — verificado que o próprio `rpm` padronizado já domina tanto a distância
euclidiana que os vizinhos mais próximos, mesmo sem filtro explícito, já pertencem quase
sempre à mesma faixa de rotação. O filtro foi mantido como salvaguarda explícita e barata
(protege contra mudanças futuras no conjunto de features), mas o teto real de acurácia
vem da sobreposição entre categorias **dentro** da mesma rotação, não de rotações
diferentes se misturando.

Decisão: aceitar 73,3% como linha de base do v1 (validado por teste estatístico, não por
um caso escolhido a dedo) e documentar como limitação conhecida — não como bug. Melhorias
futuras possíveis: (1) features específicas de frequência de defeito (BPFO/BPFI/BSF/FTF,
descritas em `Doc1.pdf`) em vez de estatísticas genéricas de amplitude; (2) modelo
supervisionado por faixa de RPM. Fora do escopo do prazo atual.
