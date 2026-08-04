# Manutenção Prescritiva

Pipeline de IA para manutenção prescritiva de máquinas rotativas: dado um novo
evento de sensor (vibração, temperatura, rpm), o sistema busca ocorrências
históricas semelhantes, identifica a categoria de defeito por similaridade e,
quando existe procedimento documentado para aquela categoria, usa um LLM para
explicar o defeito e recomendar a correção — com base exclusivamente no
documento técnico correspondente. Quando não existe documentação para o
defeito identificado, o sistema reporta isso explicitamente e permite cadastrar
um novo procedimento, em vez de inventar uma resposta.

Projeto desenvolvido para o processo seletivo de Desenvolvedor Full Stack —
I.A. e Python do SENAI SC.

## Arquitetura

Ver [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) para os diagramas de
componentes, o fluxo de decisão de um novo evento e a proposta de implantação
em ambiente industrial.

Resumo do stack:

| Camada | Escolha |
|---|---|
| Linguagem | Python |
| Interface | Streamlit |
| Orquestração LLM/RAG | LangChain |
| LLM | Groq (nuvem, primário) com fallback para Ollama (local, offline) |
| Embeddings | FastEmbed (multilíngue, local) |
| Busca vetorial | FAISS |
| Dados estruturados | SQLite |
| Extração de PDF | PyMuPDF, com fallback de OCR (RapidOCR) para documentos escaneados |

## Como rodar

### 1. Pré-requisitos

- Python 3.12+
- Uma chave de API da [Groq](https://console.groq.com) (gratuita)
- Opcional: [Ollama](https://ollama.com/download) instalado localmente, para o
  fallback offline do LLM

### 2. Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
pip install -r requirements-dev.txt   # só para rodar testes/notebooks
```

### 3. Dados

Baixe `banner.csv` e os documentos de procedimento na pasta do Google Drive
indicada no edital e coloque:

- `banner.csv` em `data/raw/` (ver `data/raw/README.md`)
- os PDFs `Doc1.pdf`...`Doc6.pdf` em `data/docs/` (já incluídos neste
  repositório, pois são pequenos)

### 4. Configuração

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/Mac
```

Edite `.env` e preencha `GROQ_API_KEY`.

### 5. Rodar o app

```bash
streamlit run src/app/streamlit_app.py
```

Na primeira execução, o app constrói automaticamente os índices de
similaridade e de documentos (pode levar 1-2 minutos, por causa do OCR do
`Doc1.pdf` — ver `docs/ARQUITETURA.md`/`notas-internas/DECISIONS.md`). Nas
execuções seguintes isso é instantâneo, pois os índices ficam em
`data/index/` (não versionado — são artefatos derivados, regeneráveis a
qualquer momento).

Abra `http://localhost:8501`. Um evento de exemplo (o mesmo do edital) já vem
pré-carregado na aba "Novo evento".

### 6. Rodar os testes

```bash
pytest tests/ -v
```

### 7. Notebook de EDA

`notebooks/01_eda.ipynb` — análise exploratória do `banner.csv` (distribuição
de defeitos, cobertura documental, separabilidade das features de sensor).

## Estrutura do repositório

```
data/
  docs/       procedimentos técnicos (PDFs) usados no RAG
  raw/        banner.csv (não versionado — ver data/raw/README.md)
docs/
  ARQUITETURA.md   arquitetura da solução e proposta de implantação
notebooks/
  01_eda.ipynb     análise exploratória
src/
  data/       carregamento e normalização dos rótulos de fault
  similarity/ busca por similaridade (features de sensor)
  rag/        ingestão de documentos e recuperação por categoria
  llm/        orquestração do LLM e a trava anti-alucinação
  app/        interface Streamlit
tests/        suíte de testes (pytest)
```

## Decisões técnicas e achados

O histórico de decisões técnicas, bugs encontrados durante o desenvolvimento e
a justificativa de cada escolha de arquitetura ficam documentados em
`notas-internas/DECISIONS.md` — não versionado neste repositório (é um diário
de bordo pessoal), mas o essencial de cada decisão está resumido em
`docs/ARQUITETURA.md`.
