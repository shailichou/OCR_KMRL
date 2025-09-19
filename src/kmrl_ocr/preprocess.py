import cv2

def preprocess_image(image_path: str):
    """
    Preprocess an image for OCR:
    - Reads the image
    - Converts to grayscale
    - Applies adaptive thresholding
    - Removes noise
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 11
    )

    
    denoised = cv2.medianBlur(thresh, 3)

    return denoised

