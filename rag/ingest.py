"""
Ingestion entrypoint: wires parse -> chunk -> embed into one run.
Each stage stays its own testable module; this is the only place that
needs to know how they connect.
"""
from pathlib import Path

from .doc_parser import convert_pdf_raw, convert_docx_raw, convert_TXT_raw
from .content_processor import markdown_to_chunks
from .vectorstore import upload_data, sparse_model, dense_model

PDF_DIR = Path("data/PDFS")
DOCX_DIR = Path("data/DOCX")
TXT_DIR = Path("data/TXT")
RAW_MD_DIR = Path("data/raw")




def run_ingestion() -> None:
    print("Starting multi-format corpus ingestion...")
    all_md_paths = []

    # 1. Parse PDFs
    pdf_paths = convert_pdf_raw(pdf_dir=PDF_DIR, output_dir=RAW_MD_DIR)
    print(f"Parsed {len(pdf_paths)} PDF(s) into markdown.")
    all_md_paths.extend(pdf_paths)

    # 2. Parse DOCX
    docx_paths = convert_docx_raw(docx_dir=DOCX_DIR, output_dir=RAW_MD_DIR)
    print(f"Parsed {len(docx_paths)} DOCX file(s) into markdown.")
    all_md_paths.extend(docx_paths)

    # 3. Parse TXT
    txt_paths = convert_TXT_raw(TXT_dir=TXT_DIR, output_dir=RAW_MD_DIR)
    print(f"Parsed {len(txt_paths)} TXT/MD file(s) into markdown.")
    all_md_paths.extend(txt_paths)

    print(f"Total markdown documents to chunk: {len(all_md_paths)}")

    # 4. Chunk
    chunks = markdown_to_chunks(RAW_MD_DIR)
    print(f"Produced {len(chunks)} chunks across all 14 clinical guideline documents.")


    texts = [doc.page_content for doc in chunks]
    #5. Embed Dense vectors
    dense_vectors = dense_model.embed_documents(texts)
    sparse_vectors = sparse_model.embed(texts)
    # 5. Build Qdrant Vector Store
    upload_data(texts, dense_vectors=dense_vectors, sparse_vectors=sparse_vectors)
    print("Ingestion complete -- multi-format vector store is ready.")


if __name__ == "__main__":
    run_ingestion()