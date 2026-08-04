"""Ingestão dos documentos de procedimento (data/docs) em um índice vetorial FAISS.

Cada chunk carrega metadados (`doc_file`, `category`) para permitir filtrar a
recuperação por documento na hora do chat — evitando que a resposta sobre um
defeito misture trechos de um procedimento de outro defeito.

Modelo de embedding multilíngue (os documentos e o chat são em português):
sentence-transformers/paraphrase-multilingual-mpnet-base-v2, via FastEmbed
(execução local em CPU, sem depender de API externa nem GPU).

Extração de texto com fallback de OCR: `Doc1.pdf` (rolamentos — a maior
categoria de defeito da base) não tem camada de texto extraível por nenhuma
biblioteca de parsing padrão (pypdf, PyMuPDF) — o conteúdo das páginas é uma
imagem. Extrair texto nativo primeiro e só recorrer a OCR (RapidOCR, local,
sem binário externo) quando a página vier vazia evita pagar o custo de OCR
nos outros 5 documentos, que são texto nativo normal.
"""

from pathlib import Path

import fitz  # PyMuPDF
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rapidocr import RapidOCR

from src.data.labels import CATEGORY_TO_DOC, load_doc_overrides, register_document_override

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DOCS_DIR = Path("data/docs")
VECTORSTORE_DIR = Path("data/index/docs_faiss")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Abaixo deste número de caracteres, assume-se que a página não tem texto
# nativo extraível (PDF escaneado/imagem) e recorre-se a OCR.
MIN_CHARS_TEXTO_NATIVO = 20

_ocr_engine: RapidOCR | None = None


def _embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def _get_ocr_engine() -> RapidOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _category_for_doc_file(doc_file: str) -> str:
    resolved = {**CATEGORY_TO_DOC, **load_doc_overrides()}
    for category, arquivo in resolved.items():
        if arquivo == doc_file:
            return category
    return "desconhecido"


def _texto_da_pagina(page: fitz.Page) -> str:
    """Texto nativo da página; recorre a OCR se a página for uma imagem."""
    texto = page.get_text()
    if len(texto.strip()) >= MIN_CHARS_TEXTO_NATIVO:
        return texto

    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    resultado = _get_ocr_engine()(pixmap.tobytes("png"))
    return "\n".join(resultado.txts) if resultado.txts else ""


def _load_pdf_as_documents(pdf_path: Path) -> list[Document]:
    with fitz.open(str(pdf_path)) as pdf:
        return [
            Document(
                page_content=_texto_da_pagina(page),
                metadata={"source": str(pdf_path), "page": numero},
            )
            for numero, page in enumerate(pdf)
        ]


def load_and_chunk(pdf_path: Path, category: str | None = None) -> list[Document]:
    """Carrega um PDF (com fallback de OCR) e o divide em chunks com metadados."""
    pages = _load_pdf_as_documents(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    categoria = category or _category_for_doc_file(pdf_path.name)
    for chunk in chunks:
        chunk.metadata["doc_file"] = pdf_path.name
        chunk.metadata["category"] = categoria
    return chunks


def build_vectorstore(docs_dir: Path = DOCS_DIR) -> FAISS:
    """Reconstrói o índice vetorial do zero a partir de todos os PDFs em docs_dir."""
    todos_os_chunks: list[Document] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        todos_os_chunks.extend(load_and_chunk(pdf_path))

    vectorstore = FAISS.from_documents(todos_os_chunks, _embeddings())
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def load_vectorstore() -> FAISS:
    return FAISS.load_local(
        str(VECTORSTORE_DIR), _embeddings(), allow_dangerous_deserialization=True
    )


def add_document(pdf_path: Path, category: str) -> None:
    """Ingesta um novo documento e o associa a uma categoria de defeito.

    Usado quando o usuário registra, via chat, o procedimento para uma
    categoria que ainda não tinha documentação (eccentric_rotor, ventoinha,
    falta_fase). Não reconstrói o índice de similaridade — quem chama isto
    deve rodar src.similarity.build.build_index() depois, para que
    has_documentation reflita a mudança nos eventos históricos.
    """
    chunks = load_and_chunk(pdf_path, category=category)
    vectorstore = load_vectorstore()
    vectorstore.add_documents(chunks)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    register_document_override(category, pdf_path.name)


def retrieve_for_category(query: str, category: str, k: int = 4) -> list[Document]:
    """Busca os trechos mais relevantes dentro do documento de uma categoria."""
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(
        query, k=k, filter={"category": category}
    )
