from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """
    Accepts the path of a PDF file, reads every page,
    combines all page text into one string, and returns it.
    """
    reader = PdfReader(file_path)
    text_content = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
            
    return "\n".join(text_content)
