def create_chunks(pages, chunk_size=500, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page_number": page_number
            })

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from document_loader import load_pdf

    pdf_path = "data/documents/sample.pdf"

    pages = load_pdf(pdf_path)

    chunks = create_chunks(pages)

    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print("Page:", chunk["page_number"])
        print(chunk["text"][:500])