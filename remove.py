import os
import nibabel as nib
import numpy as np
from glob import glob

base_dir = "D:/monai-project/Data_Train_Test"

# Folder pairs: (images, labels)
folder_pairs = [
    ("TrainVolumes", "TrainSegmentation"),
    ("TestVolumes", "TestSegmentation"),
]

removed_count = 0

for img_folder, label_folder in folder_pairs:
    img_dir = os.path.join(base_dir, img_folder)
    label_dir = os.path.join(base_dir, label_folder)
    label_files = sorted(glob(os.path.join(label_dir, "*.nii*")))

    for label_path in label_files:
        label_data = nib.load(label_path).get_fdata()
        if np.all(label_data == 0):
            base_name = os.path.splitext(os.path.splitext(os.path.basename(label_path))[0])[0]
            img_path = os.path.join(img_dir, f"{base_name}.nii.gz")
            os.remove(label_path)
            if os.path.exists(img_path):
                os.remove(img_path)
            removed_count += 1
            print(f"Removed empty pair: {base_name}")

print(f"Total empty pairs removed: {removed_count}")