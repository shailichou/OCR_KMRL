from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil

from kmrl_ocr.ocr import extract_blocks
from kmrl_ocr.utils import pdf_to_images, extract_text_from_pdf
from kmrl_ocr.exporter import save_results

app = FastAPI(title="KMRL OCR API", description="Single endpoint OCR service")

DATASET_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_document(doc_path: str, output_path="output.json"):
    pages = []

    if doc_path.endswith(".pdf"):
        digital_pages = extract_text_from_pdf(doc_path)

        if any(p["text"] for p in digital_pages):
            print("[INFO] Digital PDF detected → extracting text directly")
            for p in digital_pages:
                pages.append({
                    "page": p["page"],
                    "file": doc_path,
                    "blocks": [{
                        "text": p["text"],
                        "lang": "en",
                        "confidence": 100,
                        "bbox": None
                    }]
                })
        else:
            print("[INFO] Scanned PDF detected → running OCR")
            image_paths = pdf_to_images(doc_path)
            for idx, img in enumerate(image_paths):
                blocks = extract_blocks(img)
                pages.append({
                    "page": idx + 1,
                    "file": img,
                    "blocks": blocks
                })
    else:
        blocks = extract_blocks(doc_path)
        pages.append({"page": 1, "file": doc_path, "blocks": blocks})

    save_results(pages, output_path)
    return pages


@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or image).
    Returns the extracted text/blocks and saves JSON.
    """
    # Save uploaded file
    file_path = os.path.join(DATASET_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define output JSON path
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file.filename)[0]}.json")

    # Run OCR pipeline
    results = process_document(file_path, output_file)

    return JSONResponse(content={
        "status": "success",
        "file": file.filename,
        "output": output_file,
        "results": results
    })
