import io
from pypdf import PdfReader

def extract_text_from_pdf(file_path_or_bytes) -> str:
    """
    Accepts the path of a PDF file or bytes/stream, reads every page,
    combines all page text into one string, and returns it.
    """
    if isinstance(file_path_or_bytes, bytes):
        reader = PdfReader(io.BytesIO(file_path_or_bytes))
    else:
        reader = PdfReader(file_path_or_bytes)
        
    text_content = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
            
    return "\n".join(text_content)
