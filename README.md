Instrukcja - maski_pluc2.py

1. Pobrać dataset ze strony: https://zenodo.org/records/3723295 - np. subset0.zip + annotations.csv
2. Pliki .mhd i .raw wraz z annotations.csv umieścić w folderze "LUNA16"
  
  
Program maski_pluc2.py dokonuje segmentacji obrazów tworząc maski płuc, dodatkowo wybiera tylko te zdjęcia, na których płuca zajmują minimum 15% powierzchni. Stosuje maski na obrazach po wcześniejszej dylatacji, tak, aby wyciąć tło. Zapisuje wynikowe obrazy jako .png, maski jako .jpg, a dodatkowo maski jako .npy. Bierze pod uwagę jedynie środkową część sliceów (35, 65)[%]



p2p_test2.py
Program trenujący i wizualizujący wyniki sieci Pix2Pix wzorowanej na tym artykule: https://www.sciencedirect.com/science/article/abs/pii/S0957417422023685


![epoch_395](https://github.com/user-attachments/assets/92789200-ae22-4cca-89e3-04bb8fdaf962)
