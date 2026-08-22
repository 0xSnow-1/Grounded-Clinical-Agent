from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document


def markdown_to_chunks(md_dir: Path) -> list[Document]:

    documents = []

    # 1. Markdown files → initial Documents
    for md_path in md_dir.glob("*.md"):
        text = md_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(md_path)
                },
            )
        )

    # 2. Define which Markdown headers represent structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 3. Split by Markdown structure first
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    header_splits = []

    for document in documents:
        splits = markdown_splitter.split_text(document.page_content)

        for split in splits:
            split.metadata["source"] = document.metadata["source"]

        header_splits.extend(splits)

    # 4. Split large sections into size-controlled chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # 5. Return final chunk Documents
    return text_splitter.split_documents(header_splits)






















# def chunk_markdown_file(
#     md_path: Path,
#     chunk_size: int = 1000,
#     chunk_overlap: int = 200,
# ) -> list[Document]:
#     """Splits a single markdown file into chunks: first by header, then by size."""
#     if not md_path.is_file():
#         raise FileNotFoundError(f"File not found: {md_path}")

#     text = md_path.read_text(encoding="utf-8")

#     headers_to_split_on = [("##", "main_topic")]
#     header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
#     header_chunks = header_splitter.split_text(text)

#     # Tag every section with its source file before size-splitting -- split_documents()
#     # carries existing metadata forward onto the final chunks.
#     source_name = md_path.stem
#     for section in header_chunks:
#         section.metadata["source"] = source_name

#     size_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     return size_splitter.split_documents(header_chunks)


# def chunk_markdown_files(
#     md_paths: list[Path],
#     chunk_size: int = 1000,
#     chunk_overlap: int = 200,
# ) -> list[Document]:
#     """Chunks a batch of markdown files -- the actual shape convert_pdf_raw /
#     convert_docx_raw / convert_TXT_raw return -- and aggregates into one list."""
#     all_chunks: list[Document] = []
#     for md_path in md_paths:
#         all_chunks.extend(chunk_markdown_file(md_path, chunk_size, chunk_overlap))
#     return all_chunks


# if __name__ == "__main__":
#     from .doc_parser import convert_pdf_raw

#     md_paths = convert_pdf_raw(
#         pdf_dir=Path("data/PDFS"),
#         output_dir=Path("data/raw"),
#     )
#     chunks = chunk_markdown_files(md_paths)
#     print(f"Produced {len(chunks)} chunks from {len(md_paths)} files.")
#     if chunks:
#         print("--- sample chunk ---")
#         print(chunks[0].metadata)
#         print(chunks[0].page_content[:300])