from pdf2image import convert_from_path
import os
import shutil
from langdetect import detect
import fitz

def pdf_to_images(pdf_path, output_folder="data/temp_images", dpi=300):
    """
    Convert PDF into images (one per page).
    Returns list of image paths.
    """
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    
    images = convert_from_path(pdf_path, dpi=dpi)

    paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(path, "PNG")
        paths.append(path)
    return paths

def detect_language(text: str):
    """
    Detect language of a text block.
    Returns 'en', 'ml', or 'unknown'
    """
    try:
        return detect(text)
    except:
        return "unknown"
    
def extract_text_from_pdf(pdf_path: str):

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page": i + 1,
            "text": text.strip()
        })
    return pages

