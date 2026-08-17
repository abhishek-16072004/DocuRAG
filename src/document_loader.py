from pypdf import PdfReader


def load_pdf(file_path):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    return pages


if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    pages = load_pdf(pdf_path)

    print("PDF pages:", len(PdfReader(pdf_path).pages))
    print("Pages containing text:", len(pages))

    print("\nPage numbers containing text:")

    for page in pages:
        print(page["page_number"], end=" ")