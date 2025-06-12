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
OUT_DIR = "dataset_masks2"
os.makedirs(OUT_DIR, exist_ok=True)
MIN_WHITE_RATIO = 0.15

mhd_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mhd')]

# Wczytaj adnotacje i pogrupuj według seriesuid
annotations = pd.read_csv(ANNOT_PATH)
tumor_dict = annotations.groupby('seriesuid').apply(lambda x: x[['coordX', 'coordY', 'coordZ', 'diameter_mm']].to_dict('records')).to_dict()

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
    spacing = itk_img.GetSpacing()
    
    # Pobierz guzy dla tej serii
    tumors = tumor_dict.get(seriesuid, [])
    
    # Oblicz zajęte slice'y i pozycje guzów
    affected_slices = set()
    tumor_positions = {}
    for tumor in tumors:
        try:
            point = (tumor['coordX'], tumor['coordY'], tumor['coordZ'])
            index = itk_img.TransformPhysicalPointToIndex(point)
            z_index = int(round(index[2]))
            delta = int(0.75 * (tumor['diameter_mm'] / spacing[2]))
            
            # Zakres slice'y do pominięcia
            start = max(0, z_index - delta)
            end = min(num_slices-1, z_index + delta)
            affected_slices.update(range(start, end+1))
            
            # Zapisz pozycję guza
            if z_index not in tumor_positions:
                tumor_positions[z_index] = []
            tumor_positions[z_index].append({
                'x': int(round(index[0])),
                'y': int(round(index[1])),
                'diameter': tumor['diameter_mm']
            })
        except:
            continue

    # Utwórz foldery
    out_with = os.path.join(OUT_DIR, 'with_nodules')
    out_without = os.path.join(OUT_DIR, 'without_nodules')
    os.makedirs(out_with, exist_ok=True)
    os.makedirs(out_without, exist_ok=True)

    start_slice = int(num_slices * 0.35)
    end_slice = int(num_slices * 0.65)
    
    lung_mask = process_volume(itk_img, start_slice, end_slice)
    np.save(f'{OUT_DIR}/{seriesuid}_lung_mask_volume.npy', lung_mask)

    for slice_idx in range(num_slices):
        # Określ folder docelowy
        output_dir = out_with if slice_idx in affected_slices else out_without
        
        arr = img[slice_idx]
        arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        mask_slice = lung_mask[slice_idx]

        # Przetwarzanie maski
        mask_slice_bool = (mask_slice == 255)
        dilated_mask = binary_dilation(mask_slice_bool, iterations=2)
        dilated_mask = dilated_mask.astype(np.uint8) * 255
        
        # Wygladzanie
        kernel = np.ones((3,3), np.uint8)
        smoothed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
        blurred_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (5,5), 0)
        _, final_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        # Nanieś guzy jeśli istnieją w tym slicie
        if slice_idx in tumor_positions:
            mask_with_tumor = cv2.cvtColor(final_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for tumor in tumor_positions[slice_idx]:
                x = tumor['x']
                y = tumor['y']
                radius = int((tumor['diameter'] / spacing[0]) / 2)
                cv2.circle(mask_with_tumor, (x, y), radius, (128, 128, 128), -1)
            final_mask = cv2.cvtColor(mask_with_tumor, cv2.COLOR_BGR2GRAY)

        white_pixels = np.sum(final_mask == 255)
        total_pixels = final_mask.size
        if white_pixels / total_pixels >= MIN_WHITE_RATIO:
            # Zapisz wyniki
            masked_arr_norm = arr_norm * (final_mask / 255)
            
            plt.imsave(
                f"{output_dir}/{seriesuid}_slice_{slice_idx}.png",
                masked_arr_norm,
                cmap='gray'
            )
            plt.imsave(
                f"{output_dir}/{seriesuid}_mask_{slice_idx}.jpg",
                final_mask,
                cmap='gray',
                vmin=0,
                vmax=255
            )
            np.save(f'{output_dir}/{seriesuid}_lung_mask_{slice_idx}.npy', final_mask)

print("Przetwarzanie zakończone!")
