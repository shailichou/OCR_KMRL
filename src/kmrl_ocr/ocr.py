import pytesseract
from PIL import Image
from .preprocess import preprocess_image
from .utils import detect_language

def extract_blocks(image_path: str, lang: str = "eng+mal"):
    """
    Extract text blocks with bounding boxes, confidence, and language.
    Returns a list of dicts.
    """
    img = preprocess_image(image_path)
    pil_img = Image.fromarray(img)

    
    data = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        output_type=pytesseract.Output.DICT
    )

    blocks = []
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if word != "" and conf > 40:  
            bbox = {
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i])
            }
            lang_detected = detect_language(word)
            blocks.append({
                "text": word,
                "lang": lang_detected,
                "confidence": conf,
                "bbox": bbox
            })

    return blocks

