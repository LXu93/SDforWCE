import os
import pandas as pd
from PIL import Image
import numpy as np

def compute_highlight_ratio(image_path, intensity_threshold=200):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert to intensity
    gray = np.mean(image_np, axis=2)

    # Count highlighted pixels
    highlighted = gray > intensity_threshold
    highlight_ratio = highlighted.sum() / gray.size

    return highlight_ratio

def filter_images_by_highlight(image_folder, label="clean", ratio_threshold=0.001, intensity_threshold=200):
    results = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, fname)
            ratio = compute_highlight_ratio(image_path, intensity_threshold=intensity_threshold)
            keep = not (label == "clean" and ratio > ratio_threshold)
            results.append({
                "filename": fname,
                "highlight_ratio": round(ratio, 5),
                "keep": keep
            })
    return results

if __name__ == "__main__":
    image_folder = "D:\Lu\data_thesis\Galar_tech\small_intestine\clean"

    filtered = filter_images_by_highlight(image_folder)

    output_csv_path = "highlight_filter_result.csv"
    pd.DataFrame(filtered).to_csv(output_csv_path, index=False)
