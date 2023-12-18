import cv2
import os
import sys
import glob
from dotenv import load_dotenv
import numpy as np

def main():
    FILE_PATH = os.getenv('TARGET_DIR')
    OUTPUT_PATH = f"{os.getenv('OUTPUT_DIR')}/{os.getenv('OUTPUT_FILENAME')}"
    OUTPUT_FILE = f"{OUTPUT_PATH}.png"
    R_LAYER_FILE = f"{OUTPUT_PATH}-R.png"
    G_LAYER_FILE = f"{OUTPUT_PATH}-G.png"
    B_LAYER_FILE = f"{OUTPUT_PATH}-B.png"
    MAX_LAYER_FILE = f"{OUTPUT_PATH}-MAX.png"
    MIN_LAYER_FILE = f"{OUTPUT_PATH}-MIN.png"
    MEDIAN_LAYER_FILE = f"{OUTPUT_PATH}-MEDIAN.png"
    MEAN_MEDIAN_DIFF_LAYER_FILE = f"{OUTPUT_PATH}-MEDIAN-DIFF.png"
    SORTED_MEDIAN_LAYER_FILE = f"{OUTPUT_PATH}-SORTED.png"
    DENOISED_FILE = f"{OUTPUT_PATH}-DENOISED.png"
    print(f"Loading images from {FILE_PATH}")

    image_files = glob.glob(f"{FILE_PATH}/**/*.png", recursive=True)

    images = [cv2.imread(i) for i in image_files ]

    print(f"{len(image_files)} Images Read")
    
    mean_image = np.mean(images, axis=0)
    print(mean_image.shape)
    cv2.imwrite(OUTPUT_FILE, mean_image)
    r,g,b = cv2.split(mean_image)

    cv2.imwrite(R_LAYER_FILE, r)
    cv2.imwrite(G_LAYER_FILE, g)
    cv2.imwrite(B_LAYER_FILE, b)

    max_image = np.max(images, axis=0)
    cv2.imwrite(MAX_LAYER_FILE, max_image)
    cv2.imwrite(MIN_LAYER_FILE, np.min(images, axis=0))

    cv2.imwrite(MEDIAN_LAYER_FILE, np.median(images, axis=0))
    cv2.imwrite(SORTED_MEDIAN_LAYER_FILE, np.sort(np.median(images, axis=0), axis=0))

    cv2.imwrite(MEAN_MEDIAN_DIFF_LAYER_FILE, 4.0 * np.abs(np.mean(images, axis=0) - np.median(images, axis=0)))
    cv2.imwrite(DENOISED_FILE, cv2.fastNlMeansDenoisingColored(np.uint8(np.median(images, axis=0)), None, 15, 10, 7, 21))
if __name__ == '__main__':
    load_dotenv()
    main()
