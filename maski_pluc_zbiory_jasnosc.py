import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lungmask import LMInferer
from scipy.ndimage import binary_dilation
import cv2
import shutil
from sklearn.model_selection import train_test_split

# Konfiguracja ścieżek
DATA_DIR = "LUNA16"
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.csv")
OUT_DIR = "dataset_masks_zb"

# Parametry
MIN_WHITE_RATIO = 0.15
NODULE_VALUE = 128  # Wartość szarości dla guzka
BRIGHTNESS_THRESHOLD = 60  # NOWY: Próg średniej jasności
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Wczytanie danych
annotations = pd.read_csv(ANNOT_PATH)
mhd_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mhd')]

# Inicjalizacja modelu segmentacji płuc
inferer = LMInferer(modelname='R231')

def create_dataset_splits(mhd_files, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Podział pacjentów na zbiory treningowy, walidacyjny i testowy
    """
    np.random.seed(random_state)
    mhd_files = np.array(mhd_files)
    np.random.shuffle(mhd_files)
    
    n_total = len(mhd_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = mhd_files[:n_train]
    val_files = mhd_files[n_train:n_train + n_val]
    test_files = mhd_files[n_train + n_val:]
    
    return train_files, val_files, test_files

def create_directory_structure(base_dir, splits):
    """
    Tworzenie struktury folderów dla każdego zbioru
    """
    for split_name in splits:
        # Tworzenie głównych folderów dla każdego zbioru
        split_dir = os.path.join(base_dir, split_name)
        nodules_dir = os.path.join(split_dir, "with_nodules")
        clean_dir = os.path.join(split_dir, "clean")
        
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(nodules_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

def calculate_lung_brightness(original_image, lung_mask):
    """
    NOWA FUNKCJA: Obliczanie średniej jasności w obszarze płuc na oryginalnym obrazie
    """
    # Tworzenie maski binarnej płuc
    lung_binary_mask = (lung_mask > 0)
    
    # Sprawdzenie czy maska zawiera jakiekolwiek piksele płuc
    if not np.any(lung_binary_mask):
        return 0
    
    # Obliczenie średniej jasności tylko w obszarze płuc
    lung_pixels = original_image[lung_binary_mask]
    mean_brightness = np.mean(lung_pixels)
    
    return mean_brightness

def world_to_voxel(coord, origin, spacing):
    """Konwersja współrzędnych świata do voxeli"""
    return np.round((np.array(coord) - origin) / spacing).astype(int)

def process_volume(itk_img, start_slice, end_slice):
    """Segmentacja płuc w całym wolumenie"""
    segmentation = inferer.apply(itk_img)
    mask_3d = (segmentation > 0).astype(np.uint8) * 255
    mask_3d[:start_slice, :, :] = 0
    mask_3d[end_slice:, :, :] = 0
    return mask_3d

def create_nodule_mask(shape, x_idx, y_idx, z_idx, radius_mm, spacing):
    """Tworzenie maski guzka"""
    nodule_mask = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Konwersja promienia z mm na voxele
    radius_voxel_z = radius_mm / spacing[0]
    radius_voxel_y = radius_mm / spacing[1]
    radius_voxel_x = radius_mm / spacing[2]
    
    # Tworzenie elipsoidy
    mask = ((z - z_idx) / radius_voxel_z) ** 2 + \
           ((y - y_idx) / radius_voxel_y) ** 2 + \
           ((x - x_idx) / radius_voxel_x) ** 2 <= 1
    
    nodule_mask[mask] = 255
    return nodule_mask

def create_red_dot_visualization(arr_norm, nodule_positions, spacing):
    """Tworzenie obrazu RGB z czerwonymi kropkami w miejscach guzków"""
    # Konwersja obrazu szarości na RGB
    arr_rgb = np.stack([arr_norm]*3, axis=-1)
    
    for x_idx, y_idx, diameter_mm in nodule_positions:
        # Sprawdzenie czy pozycja jest w granicach obrazu
        if 0 <= y_idx < arr_rgb.shape[0] and 0 <= x_idx < arr_rgb.shape[1]:
            # Tworzenie maski koła dla czerwonej kropki
            rr, cc = np.ogrid[:arr_rgb.shape[0], :arr_rgb.shape[1]]
            radius_pixels = max(3, int((diameter_mm * 1.5 / 2) / spacing[1]))  # minimum 3 piksele
            mask = (rr - y_idx)**2 + (cc - x_idx)**2 <= radius_pixels**2
            
            # Zaznaczenie czerwoną kropką
            arr_rgb[mask, 0] = np.minimum(arr_rgb[mask, 0] + 0.2, 1.0)  # Zwiększenie czerwonego kanału
            # arr_rgb[mask, 1] = arr_rgb[mask, 1] * 0.3  # Zmniejszenie zielonego kanału
            # arr_rgb[mask, 2] = arr_rgb[mask, 2] * 0.3  # Zmniejszenie niebieskiego kanału
    
    return arr_rgb

def process_patient(mhd_file, split_name, base_out_dir):
    """
    Przetwarzanie pojedynczego pacjenta i zapisanie do odpowiedniego zbioru
    """
    seriesuid = mhd_file.replace('.mhd', '')
    mhd_path = os.path.join(DATA_DIR, mhd_file)
    
    # Określenie folderów docelowych dla tego zbioru
    split_dir = os.path.join(base_out_dir, split_name)
    out_dir_nodules = os.path.join(split_dir, "with_nodules")
    out_dir_clean = os.path.join(split_dir, "clean")
    
    # Wczytanie obrazu
    itk_img = sitk.ReadImage(mhd_path)
    img = sitk.GetArrayFromImage(itk_img)  # [slices, height, width]
    origin = np.array(itk_img.GetOrigin())[::-1]  # Z, Y, X
    spacing = np.array(itk_img.GetSpacing())[::-1]  # Z, Y, X
    num_slices = img.shape[0]
    
    # Określenie zakresu płuc
    start_slice = int(num_slices * 0.35)
    end_slice = int(num_slices * 0.65)
    
    # Segmentacja płuc
    lung_mask = process_volume(itk_img, start_slice, end_slice)
    
    # Wyszukanie guzków dla tego seriesuid
    nodule_rows = annotations[annotations['seriesuid'] == seriesuid]
    
    # Tworzenie maski guzków 3D
    nodule_mask_3d = np.zeros_like(lung_mask)
    
    # Tylko slice'y z centrum guzka
    center_slices_with_nodules = set()
    nodule_positions_per_slice = {}  # słownik: slice_idx -> lista pozycji guzków
    
    # Przetwarzanie każdego guzka
    for _, row in nodule_rows.iterrows():
        # Konwersja współrzędnych świata na voxele
        z_idx = int(round((row['coordZ'] - origin[0]) / spacing[0]))
        y_idx = int(round((row['coordY'] - origin[1]) / spacing[1]))
        x_idx = int(round((row['coordX'] - origin[2]) / spacing[2]))
        
        # Promień guzka (z współczynnikiem 1.5)
        radius_mm = (row['diameter_mm'] * 1.5) / 2
        
        # Dodanie tylko slice'a z centrum guzka
        if 0 <= z_idx < num_slices:
            center_slices_with_nodules.add(z_idx)
            if z_idx not in nodule_positions_per_slice:
                nodule_positions_per_slice[z_idx] = []
            nodule_positions_per_slice[z_idx].append((x_idx, y_idx, row['diameter_mm']))
        
        # Tworzenie maski guzka dla tego konkretnego guzka
        nodule_mask = create_nodule_mask(img.shape, x_idx, y_idx, z_idx, radius_mm, spacing)
        
        # Dodanie maski guzka do ogólnej maski guzków
        nodule_mask_3d = np.maximum(nodule_mask_3d, nodule_mask)
    
    # Zapisanie całych wolumenów masek w folderze zbioru
    # np.save(f'{split_dir}/{seriesuid}_lung_mask_volume.npy', lung_mask)
    # np.save(f'{split_dir}/{seriesuid}_nodule_mask_volume.npy', nodule_mask_3d)
    
    # Liczniki slice'ów dla statystyk
    nodule_slices_saved = 0
    clean_slices_saved = 0
    brightness_filtered_count = 0  # NOWY: Licznik odfiltrowanych przez jasność
    
    # Przetwarzanie każdego slice'a
    for slice_idx in range(num_slices):
        # Pobranie obrazu i normalizacja
        arr = img[slice_idx]  # Oryginalny obraz (nieznormalizowany)
        arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        
        # Pobranie maski płuc dla tego slice'a
        mask_slice = lung_mask[slice_idx]
        
        # Dylatacja maski płuc
        mask_slice_bool = (mask_slice == 255)
        dilated_mask = binary_dilation(mask_slice_bool, iterations=2)
        dilated_mask = dilated_mask.astype(np.uint8) * 255
        
        # Wygładzanie maski
        kernel = np.ones((3,3), np.uint8)
        smoothed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
        
        # Rozmycie Gaussowskie
        blurred_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (5,5), 0)
        _, blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Sprawdzenie czy maska spełnia kryterium MIN_WHITE_RATIO
        white_pixels = np.sum(dilated_mask == 255)
        total_pixels = dilated_mask.size
        
        if white_pixels / total_pixels >= MIN_WHITE_RATIO:

            
            # Pobranie maski guzka dla tego slice'a
            nodule_mask_slice = nodule_mask_3d[slice_idx]
            
            # Tworzenie maski płuc z guzkiem (255 dla płuc, 128 dla guzka)
            combined_mask = dilated_mask.copy()
            combined_mask[nodule_mask_slice == 255] = NODULE_VALUE
            
            # Zastosowanie maski na znormalizowanym obrazie
            masked_arr_norm = arr_norm * (dilated_mask / 255)
            
            # NOWA FUNKCJONALNOŚĆ: Sprawdzenie średniej jasności w obszarze płuc
            lung_brightness = calculate_lung_brightness(masked_arr_norm, dilated_mask) * 255

            # print("Jasonsc: ", lung_brightness)
            
            # Pomiń slice jeśli średnia jasność przekracza próg
            if lung_brightness > BRIGHTNESS_THRESHOLD:
                brightness_filtered_count += 1
                continue

            # Slice'y z centrum guzka do with_nodules
            if slice_idx in center_slices_with_nodules:
                target_dir = out_dir_nodules
                
                # Zapisanie obrazu z maską
                plt.imsave(
                    f"{target_dir}/{seriesuid}_slice_{slice_idx}.png",
                    masked_arr_norm,
                    cmap='gray'
                )
                
                # Zapisanie maski płuc (z guzkiem)
                plt.imsave(
                    f"{target_dir}/{seriesuid}_mask_{slice_idx}.jpg",
                    combined_mask,
                    cmap='gray',
                    vmin=0,
                    vmax=255
                )
                
                # Zapisanie maski płuc w formacie NPY
                np.save(f'{target_dir}/{seriesuid}_lung_mask_{slice_idx}.npy', combined_mask)
                
                # Zapisanie osobnej maski guzka
                np.save(f'{target_dir}/{seriesuid}_nodule_mask_{slice_idx}.npy', nodule_mask_slice)
                
                # Tworzenie i zapisanie obrazu z czerwonymi kropkami
                nodule_positions = nodule_positions_per_slice[slice_idx]
                arr_rgb_with_dots = create_red_dot_visualization(masked_arr_norm, nodule_positions, spacing)
                
                # Zapisanie obrazu z czerwonymi kropkami w formacie JPG
                plt.imsave(
                    f"{target_dir}/{seriesuid}_red_dots_{slice_idx}.jpg",
                    arr_rgb_with_dots,
                    format='jpeg',
                    dpi=150
                )
                
                nodule_slices_saved += 1
            
            # Slice'y bez centrum guzka i bez żadnej części guzka do clean
            elif slice_idx not in center_slices_with_nodules and np.sum(nodule_mask_slice) == 0:
                target_dir = out_dir_clean
                
                # Zapisanie obrazu z maską (bez guzka)
                plt.imsave(
                    f"{target_dir}/{seriesuid}_slice_{slice_idx}.png",
                    masked_arr_norm,
                    cmap='gray'
                )
                
                # Zapisanie maski płuc (bez guzka)
                plt.imsave(
                    f"{target_dir}/{seriesuid}_mask_{slice_idx}.jpg",
                    dilated_mask,  # Używamy oryginalnej maski bez guzka
                    cmap='gray',
                    vmin=0,
                    vmax=255
                )
                
                # Zapisanie maski płuc w formacie NPY
                np.save(f'{target_dir}/{seriesuid}_lung_mask_{slice_idx}.npy', dilated_mask)
                
                clean_slices_saved += 1
    
    return nodule_slices_saved, clean_slices_saved, brightness_filtered_count

# GŁÓWNY PRZEPŁYW PROGRAMU

# 1. Podział pacjentów na zbiory
print("Podział pacjentów na zbiory...")
train_files, val_files, test_files = create_dataset_splits(
    mhd_files, 
    train_ratio=TRAIN_RATIO, 
    val_ratio=VAL_RATIO, 
    test_ratio=TEST_RATIO
)

print(f"Liczba pacjentów:")
print(f"  - Treningowy: {len(train_files)}")
print(f"  - Walidacyjny: {len(val_files)}")
print(f"  - Testowy: {len(test_files)}")

# 2. Tworzenie struktury folderów
splits = ['train', 'val', 'test']
create_directory_structure(OUT_DIR, splits)

# 3. Przetwarzanie każdego zbioru
dataset_splits = {
    'train': train_files,
    'val': val_files,
    'test': test_files
}

total_stats = {
    'train': {'nodule_slices': 0, 'clean_slices': 0, 'brightness_filtered': 0, 'patients': 0},
    'val': {'nodule_slices': 0, 'clean_slices': 0, 'brightness_filtered': 0, 'patients': 0},
    'test': {'nodule_slices': 0, 'clean_slices': 0, 'brightness_filtered': 0, 'patients': 0}
}

for split_name, files_list in dataset_splits.items():
    print(f"\nPrzetwarzanie zbioru {split_name.upper()}...")
    
    for i, mhd_file in enumerate(files_list, 1):
        print(f"  Przetwarzanie pacjenta {i}/{len(files_list)}: {mhd_file}")
        
        try:
            nodule_count, clean_count, brightness_filtered = process_patient(mhd_file, split_name, OUT_DIR)
            total_stats[split_name]['nodule_slices'] += nodule_count
            total_stats[split_name]['clean_slices'] += clean_count
            total_stats[split_name]['brightness_filtered'] += brightness_filtered
            total_stats[split_name]['patients'] += 1
            
            # NOWY: Informacja o odfiltrowanych slice'ach
            if brightness_filtered > 0:
                print(f"    Odfiltrowano {brightness_filtered} slice'ów ze względu na jasność > {BRIGHTNESS_THRESHOLD}")
            
        except Exception as e:
            print(f"    BŁĄD przy przetwarzaniu {mhd_file}: {str(e)}")
            continue

# 4. Wyświetlenie statystyk końcowych
print("\n" + "="*60)
print("STATYSTYKI KOŃCOWE")
print("="*60)

for split_name in splits:
    stats = total_stats[split_name]
    total_processed = stats['nodule_slices'] + stats['clean_slices'] + stats['brightness_filtered']
    print(f"\nZbiór {split_name.upper()}:")
    print(f"  - Liczba pacjentów: {stats['patients']}")
    print(f"  - Slice'y z guzkami: {stats['nodule_slices']}")
    print(f"  - Slice'y czyste: {stats['clean_slices']}")
    print(f"  - Odfiltrowane przez jasność: {stats['brightness_filtered']}")
    print(f"  - Łącznie zapisanych: {stats['nodule_slices'] + stats['clean_slices']}")
    print(f"  - Łącznie przetworzonych: {total_processed}")

# Zapisanie informacji o podziale do plików
for split_name, files_list in dataset_splits.items():
    with open(f'{OUT_DIR}/{split_name}_patients.txt', 'w') as f:
        for mhd_file in files_list:
            f.write(f"{mhd_file.replace('.mhd', '')}\n")

print(f"\nSegmentacja i zapis masek zakończone!")
print(f"Próg jasności: {BRIGHTNESS_THRESHOLD}")
print(f"Pliki zostały zapisane w folderze: {OUT_DIR}")
