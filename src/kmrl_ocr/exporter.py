import json
import os

def save_results(results, output_path="output.json"):
    """
    Save OCR results into a JSON file.

    results: list of dicts (one per page), e.g.
    [
      {
        "page": 1,
        "file": "data/temp_images/page_1.png",
        "blocks": [
          {"text": "KMRL", "lang": "en", "confidence": 95, "bbox": {...}},
          {"text": "ക ൊച്ചി കെട്രൊ കെയിൽ", "lang": "ml", "confidence": 88, "bbox": {...}}
        ]
      }
    ]
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Results saved → {output_path}")
