import cv2
import os
import sys
import glob
from dotenv import load_dotenv
import numpy as np


def hslOperations(output_path, raw_images):

    output = f"{output_path}_hsl"
    AVERAGED_HSL = f"{output}.png"
    MEDIAN_HSV = f"{output}_median.png"
    AVERAGE_HUE = f"{output}_hue_sqaure.png"
    hsl_images = [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in raw_images]
    mean_image = np.float32(np.mean(hsl_images, axis=0))
    h, s, v = cv2.split(mean_image)
    mean_hue = np.mean(h)
    mean_image = cv2.cvtColor(mean_image, cv2.COLOR_HSV2RGB)

    median_image = np.float32(np.median(hsl_images, axis=0))
    median_image = cv2.cvtColor(mean_image, cv2.COLOR_HSV2RGB)

    print(f"Writing {AVERAGED_HSL}")
    print(mean_image.shape)
    cv2.imwrite(AVERAGED_HSL, mean_image)
    print(f"Writing {MEDIAN_HSV}")
    cv2.imwrite(MEDIAN_HSV, median_image)
    print(f"Average Hue {mean_hue}")

    # Build Hue Space Image
    s_gradient = np.tile(np.linspace(0,255,256,dtype=np.uint8), (256,1))
    v_gradient = np.tile(np.linspace(0,255,256,dtype=np.uint8).reshape(-1,1), (1,256))
    h_overlay = np.full((256,256), mean_hue,dtype=np.uint8)
    mean_hue_img = cv2.merge([h_overlay, s_gradient, v_gradient])
    cv2.imwrite(AVERAGE_HUE, cv2.cvtColor(mean_hue_img, cv2.COLOR_HSV2RGB))



def main():

    FILE_PATH = os.getenv('TARGET_DIR')
    OUTPUT_PATH = f"{os.getenv('OUTPUT_DIR')}/{os.getenv('OUTPUT_FILENAME')}"
    OUTPUT_FILE = f"{OUTPUT_PATH}.png"
    R_LAYER_FILE = f"{OUTPUT_PATH}-R.png"
    G_LAYER_FILE = f"{OUTPUT_PATH}-G.png"
    B_LAYER_FILE = f"{OUTPUT_PATH}-B.png"
    A_LAYER_FILE = f"{OUTPUT_PATH}-A.png"
    MAX_LAYER_FILE = f"{OUTPUT_PATH}-MAX.png"
    MIN_LAYER_FILE = f"{OUTPUT_PATH}-MIN.png"
    MEDIAN_LAYER_FILE = f"{OUTPUT_PATH}-MEDIAN.png"
    MEAN_MEDIAN_DIFF_LAYER_FILE = f"{OUTPUT_PATH}-MEDIAN-DIFF.png"
    SORTED_MEDIAN_LAYER_FILE = f"{OUTPUT_PATH}-SORTED.png"
    DENOISED_FILE = f"{OUTPUT_PATH}-DENOISED.png"
    print(f"Loading images from {FILE_PATH}")

    image_files = glob.glob(f"{FILE_PATH}/**/*.png", recursive=True)

    raw_images = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in image_files ]

    print(f"{len(image_files)} Images Read")
    # Add Alpha channel if not pressent
    images = []
    for i in raw_images:
        if np.shape(i)[-1] == 3:
            #print(np.shape(i), end='-->')
            i = np.dstack((i, np.full((np.shape(i)[0], np.shape(i)[1]), 255)))
            #print(np.shape(i))
        images.append(i)
        #print(np.mean(i[:,:,3]))
    mean_image = np.mean(images, axis=0)
    print(mean_image.shape)
    cv2.imwrite(OUTPUT_FILE, mean_image)
    r,g,b,a = cv2.split(mean_image)

    cv2.imwrite(R_LAYER_FILE, r)
    cv2.imwrite(G_LAYER_FILE, g)
    cv2.imwrite(B_LAYER_FILE, b)
    cv2.imwrite(A_LAYER_FILE, a)

    max_image = np.max(images, axis=0)
    cv2.imwrite(MAX_LAYER_FILE, max_image)
    cv2.imwrite(MIN_LAYER_FILE, np.min(images, axis=0))

    cv2.imwrite(MEDIAN_LAYER_FILE, np.median(images, axis=0))
    cv2.imwrite(SORTED_MEDIAN_LAYER_FILE, np.sort(np.median(images, axis=0), axis=0))

    cv2.imwrite(MEAN_MEDIAN_DIFF_LAYER_FILE, 4.0 * np.abs(np.mean(images, axis=0) - np.median(images, axis=0)))
    cv2.imwrite(DENOISED_FILE, cv2.fastNlMeansDenoisingColored(np.uint8(np.median(images, axis=0)), None, 15, 10, 7, 21))

    hslOperations(OUTPUT_PATH, raw_images)


if __name__ == '__main__':
    load_dotenv()
    main()
