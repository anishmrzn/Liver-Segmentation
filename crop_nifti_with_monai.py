import os
import nibabel as nib
import numpy as np
from glob import glob

# Set your directories
images_dir = "D:/monai-project/liver_dataset/imagesTr"
labels_dir = "D:/monai-project/liver_dataset/labelsTr"
output_images_dir = "D:/monai-project/liver_dataset/cropped/images"
output_labels_dir = "D:/monai-project/liver_dataset/cropped/labels"

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Find minimum number of slices across all images
min_slices = None
image_files = sorted(glob(os.path.join(images_dir, "*.nii*")))
for img_path in image_files:
    img = nib.load(img_path)
    num_slices = img.shape[2]
    if min_slices is None or num_slices < min_slices:
        min_slices = num_slices

print(f"Minimum number of slices found: {min_slices}")

# Crop each image and label into non-overlapping chunks of min_slices
for img_path in image_files:
    base_name = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]
    label_path = os.path.join(labels_dir, base_name + ".nii.gz")
    if not os.path.exists(label_path):
        print(f"Label not found for {base_name}, skipping.")
        continue

    img = nib.load(img_path)
    label = nib.load(label_path)
    img_data = img.get_fdata()
    label_data = label.get_fdata()

    total_slices = img_data.shape[2]
    num_crops = total_slices // min_slices

    for i in range(num_crops):
        start = i * min_slices
        end = start + min_slices
        img_crop = img_data[:, :, start:end]
        label_crop = label_data[:, :, start:end]

        # Save cropped image
        img_crop_nii = nib.Nifti1Image(img_crop, img.affine, img.header)
        img_crop_path = os.path.join(output_images_dir, f"{base_name}_crop{i}.nii.gz")
        nib.save(img_crop_nii, img_crop_path)

        # Save cropped label
        label_crop_nii = nib.Nifti1Image(label_crop, label.affine, label.header)
        label_crop_path = os.path.join(output_labels_dir, f"{base_name}_crop{i}.nii.gz")
        nib.save(label_crop_nii, label_crop_path)

        print(f"Saved crop {i} for {base_name}")

print("Cropping complete.")


