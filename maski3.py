import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lungmask import LMInferer
from scipy.ndimage import binary_dilation
import cv2

DATA_DIR = "LUNA16"
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.csv")
OUT_DIR = "dataset_masks"
os.makedirs(OUT_DIR, exist_ok=True)
MIN_WHITE_RATIO = 0.15

annotations = pd.read_csv(ANNOT_PATH)
mhd_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mhd')]

inferer = LMInferer(modelname='R231')

def process_volume(itk_img, start_slice, end_slice):
    segmentation = inferer.apply(itk_img)
    mask_3d = (segmentation > 0).astype(np.uint8) * 255
    mask_3d[:start_slice, :, :] = 0
    mask_3d[end_slice:, :, :] = 0
    return mask_3d

for mhd_file in mhd_files:
    seriesuid = mhd_file.replace('.mhd', '')
    mhd_path = os.path.join(DATA_DIR, mhd_file)
    itk_img = sitk.ReadImage(mhd_path)
    img = sitk.GetArrayFromImage(itk_img)
    num_slices = img.shape[0]

    start_slice = int(num_slices * 0.35)
    end_slice = int(num_slices * 0.65)

    lung_mask = process_volume(itk_img, start_slice, end_slice)
    np.save(f'{OUT_DIR}/{seriesuid}_lung_mask_volume.npy', lung_mask)

    for slice_idx in range(num_slices):
        arr = img[slice_idx]
        arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        mask_slice = lung_mask[slice_idx]

        # Przeprowadź dylatację maski
        mask_slice_bool = (mask_slice == 255)
        dilated_mask = binary_dilation(mask_slice_bool, iterations=2)
        dilated_mask = dilated_mask.astype(np.uint8) * 255

        kernel = np.ones((3,3), np.uint8)
        smoothed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)

        # 2. Rozmycie Gaussowskie (opcjonalnie)
        blurred_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (5,5), 0)
        # Możesz ponownie progować, jeśli chcesz maskę binarną
        _, blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        white_pixels = np.sum(dilated_mask == 255)
        total_pixels = dilated_mask.size
        if white_pixels / total_pixels >= MIN_WHITE_RATIO:
            # Zastosuj dylatowaną maskę na znormalizowanym obrazie
            masked_arr_norm = arr_norm * (dilated_mask / 255)
            plt.imsave(
                f"{OUT_DIR}/{seriesuid}_slice_{slice_idx}.png",
                masked_arr_norm,
                cmap='gray'
            )
            plt.imsave(
                f"{OUT_DIR}/{seriesuid}_mask_{slice_idx}.jpg",
                dilated_mask,
                cmap='gray',
                vmin=0,
                vmax=255
            )
            np.save(f'{OUT_DIR}/{seriesuid}_lung_mask_{slice_idx}.npy', dilated_mask)

print("Segmentacja i zapis masek zakończone!")
