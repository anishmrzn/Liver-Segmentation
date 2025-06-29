import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import io, transform, morphology, measure
import nibabel as nib
from tqdm import tqdm


TEST_VOLUMES_PATH = 'D:/monai-project/Data_Train_Test/TestVolumes'
TEST_SEGMENTATION_PATH = 'D:/monai-project/Data_Train_Test/TestSegmentation'
OUTPUT_PLOTS_PATH = 'D:/monai-project/segmentation_plots_totalseg_prompt_volume_metrics'
TOTAL_SEG_OUTPUT_PATH = 'D:/monai-project/totalseg_temp_output' 

MEDSAM_CKPT_PATH = 'medsam_vit_b.pth'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)
os.makedirs(TOTAL_SEG_OUTPUT_PATH, exist_ok=True)
print(f"Segmentation plots will be saved to: {OUTPUT_PLOTS_PATH}")
print(f"TotalSegmentator outputs will be loaded from: {TOTAL_SEG_OUTPUT_PATH}") # Updated message

MIN_COMPONENT_AREA = 500
MORPH_OPEN_DISK = 2
MORPH_CLOSE_DISK = 5


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, color='blue', lw=2):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=lw))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return 1.0
    return (2.0 * intersection) / sum_masks

def jaccard_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0
    return intersection / union

def get_bbox_from_mask(mask_slice, min_comp_area, padding_percentage=0.10):
    if np.sum(mask_slice) == 0:
        return None, None

    H, W = mask_slice.shape

    labels = measure.label(mask_slice)
    if labels.max() == 0:
        return None, None

    props = measure.regionprops(labels)
    largest_component_mask = np.zeros_like(mask_slice)
    largest_area = 0
    largest_bbox = None

    for prop in props:
        if prop.area > largest_area and prop.area >= min_comp_area:
            largest_area = prop.area
            largest_component_mask = (labels == prop.label).astype(np.uint8)
            min_row, min_col, max_row, max_col = prop.bbox

            box_width = max_col - min_col
            box_height = max_row - min_row

            pad_x = int(box_width * padding_percentage)
            pad_y = int(box_height * padding_percentage)

            min_col_padded = min_col - pad_x
            min_row_padded = min_row - pad_y
            max_col_padded = max_col + pad_x
            max_row_padded = max_row + pad_y

            min_col_final = max(0, min_col_padded)
            min_row_final = max(0, min_row_padded)
            max_col_final = min(W, max_col_padded)
            max_row_final = min(H, max_row_padded)

            if max_col_final <= min_col_final or max_row_final <= min_row_final:
                continue

            largest_bbox = [min_col_final, min_row_final, max_col_final, max_row_final]

    if largest_area == 0:
        return None, None

    return largest_component_mask, largest_bbox

def normalize_image(image_slice):
    min_val = np.min(image_slice)
    max_val = np.max(image_slice)
    if max_val - min_val == 0:
        return np.zeros_like(image_slice, dtype=np.uint8)
    normalized_slice = 255 * (image_slice - min_val) / (max_val - min_val)
    return normalized_slice.astype(np.uint8)

if __name__ == '__main__':
    print("Loading MedSAM model...")
    if not os.path.exists(MEDSAM_CKPT_PATH):
        import urllib.request
        model_url = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
        print(f"Downloading MedSAM model from {model_url} to {MEDSAM_CKPT_PATH}...")
        urllib.request.urlretrieve(model_url, MEDSAM_CKPT_PATH)
        print("Download complete.")

    medsam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    print("MedSAM model loaded successfully.")

    all_volume_dice_scores = []
    all_volume_jaccard_scores = []
    processed_volume_count = 0

    volume_files = sorted([f for f in os.listdir(TEST_VOLUMES_PATH) if f.endswith('.nii.gz')])

    # REMOVED THE TOTALSEGMENTATOR EXECUTION LOOP
    print("\n--- Skipping TotalSegmentator execution (assuming pre-generated outputs exist) ---")


    print("\n--- Starting MedSAM Inference with TotalSegmentator Prompts ---")

    for vol_fname in tqdm(volume_files, desc="Processing Volumes"):
        vol_path = os.path.join(TEST_VOLUMES_PATH, vol_fname)
        gt_seg_fname = vol_fname.replace('volume', 'segmentation').replace('img', 'label')
        if not gt_seg_fname.endswith('.nii.gz'):
             gt_seg_fname = os.path.splitext(gt_seg_fname)[0] + '.nii.gz'

        gt_seg_path = os.path.join(TEST_SEGMENTATION_PATH, gt_seg_fname)

        # Ensure base_vol_name correctly points to the sub-directory within TOTAL_SEG_OUTPUT_PATH
        # It needs to strip both '.nii' and '.gz' extensions.
        base_vol_name = os.path.splitext(os.path.splitext(vol_fname)[0])[0]
        total_seg_output_dir = os.path.join(TOTAL_SEG_OUTPUT_PATH, base_vol_name)
        total_seg_liver_path = os.path.join(total_seg_output_dir, "liver.nii.gz")

        if not os.path.exists(gt_seg_path):
            print(f"Warning: Corresponding ground truth segmentation not found for {vol_fname}. Skipping.")
            continue
        # Crucial check: make sure the specific liver segmentation file exists
        if not os.path.exists(total_seg_liver_path):
            print(f"Warning: TotalSegmentator liver output for {vol_fname} not found at {total_seg_liver_path}. Skipping.")
            continue

        try:
            nifti_vol = nib.load(vol_path).get_fdata()
            nifti_gt_seg = nib.load(gt_seg_path).get_fdata()
            nifti_total_seg_liver = nib.load(total_seg_liver_path).get_fdata()

            if not (nifti_vol.shape == nifti_gt_seg.shape == nifti_total_seg_liver.shape):
                print(f"Warning: Shape mismatch for {vol_fname} among volume, GT, and TotalSeg outputs. Skipping.")
                continue

            num_slices = nifti_vol.shape[2]
            print(f"\nProcessing {vol_fname} with {num_slices} slices...")

            medsam_seg_volume = np.zeros_like(nifti_gt_seg, dtype=np.uint8)

            representative_slice_idx = -1
            for s_offset in range(num_slices // 2, -1, -1):
                mid_slice = num_slices // 2
                current_idx = mid_slice + s_offset
                if current_idx < num_slices and np.sum(nifti_gt_seg[:, :, current_idx]) > 0:
                    representative_slice_idx = current_idx
                    break
                current_idx = mid_slice - s_offset
                if current_idx >= 0 and np.sum(nifti_gt_seg[:, :, current_idx]) > 0:
                    representative_slice_idx = current_idx
                    break

            if representative_slice_idx == -1:
                print(f"Warning: No ground truth liver found in {vol_fname}. Cannot select representative slice for plot. Skipping plots.")

            for s_idx in range(num_slices):
                image_slice_hu = nifti_vol[:, :, s_idx]
                gt_mask_slice = nifti_gt_seg[:, :, s_idx].astype(np.uint8)
                total_seg_liver_slice = nifti_total_seg_liver[:, :, s_idx].astype(np.uint8)

                prompt_mask, prompt_box = get_bbox_from_mask(
                    total_seg_liver_slice, MIN_COMPONENT_AREA, padding_percentage=0.10
                )

                medsam_seg_slice = np.zeros_like(gt_mask_slice, dtype=np.uint8)
                if prompt_box is None:
                    pass
                else:
                    img_display = normalize_image(image_slice_hu)
                    H, W = img_display.shape[:2]

                    img_3c = np.repeat(img_display[:, :, None], 3, axis=-1)
                    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
                    img_1024 = (img_1024 - img_1024.min()) / np.clip(
                        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                    )
                    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_embedding = medsam_model.image_encoder(img_1024_tensor)

                    box_np_for_medsam = np.array([prompt_box])
                    box_1024_for_medsam = box_np_for_medsam / np.array([W, H, W, H]) * 1024

                    medsam_seg_slice = medsam_inference(medsam_model, image_embedding, box_1024_for_medsam, H, W)

                medsam_seg_volume[:, :, s_idx] = medsam_seg_slice

                if s_idx == representative_slice_idx:
                    img_display_for_plot = normalize_image(image_slice_hu)
                    current_dice = dice_score(medsam_seg_slice, gt_mask_slice)
                    current_jaccard = jaccard_score(medsam_seg_slice, gt_mask_slice)

                    fig_seg, ax_seg = plt.subplots(1, 4, figsize=(24, 6))
                    fig_seg.suptitle(f"{os.path.splitext(vol_fname)[0]} - Slice {s_idx} (Rep. Plot) | Dice: {current_dice:.4f}, Jaccard: {current_jaccard:.4f}", fontsize=12)

                    ax_seg[0].imshow(img_display_for_plot, cmap='gray')
                    if prompt_box is not None:
                        show_box(prompt_box, ax_seg[0], color='red')
                    ax_seg[0].set_title('Original Image + TS-Derived Prompt Box')
                    ax_seg[0].axis('off')

                    ax_seg[1].imshow(img_display_for_plot, cmap='gray')
                    show_mask(total_seg_liver_slice, ax_seg[1], random_color=False)
                    ax_seg[1].set_title('TotalSegmentator Liver Mask (Prompt)')
                    ax_seg[1].axis('off')

                    ax_seg[2].imshow(img_display_for_plot, cmap='gray')
                    show_mask(gt_mask_slice, ax_seg[2], random_color=False)
                    ax_seg[2].set_title('Ground Truth Liver')
                    ax_seg[2].axis('off')

                    ax_seg[3].imshow(img_display_for_plot, cmap='gray')
                    show_mask(medsam_seg_slice, ax_seg[3], random_color=False)
                    ax_seg[3].set_title('MedSAM Segmentation')
                    ax_seg[3].axis('off')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.9])

                    plot_filename = f"{os.path.splitext(vol_fname)[0].replace('.nii','')}_slice_{s_idx:04d}_volume_rep_plot.png"
                    plot_filepath = os.path.join(OUTPUT_PLOTS_PATH, plot_filename)
                    plt.savefig(plot_filepath, bbox_inches='tight', dpi=100)
                    plt.close(fig_seg)

            volume_dice = dice_score(medsam_seg_volume, nifti_gt_seg)
            volume_jaccard = jaccard_score(medsam_seg_volume, nifti_gt_seg)

            all_volume_dice_scores.append(volume_dice)
            all_volume_jaccard_scores.append(volume_jaccard)
            processed_volume_count += 1

            # Adjust print statement to remove .nii.gz extension for cleaner output
            print(f"   Volume {os.path.splitext(os.path.splitext(vol_fname)[0])[0]}: Dice = {volume_dice:.4f}, Jaccard = {volume_jaccard:.4f}")

        except Exception as e:
            print(f"Error processing {vol_fname}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n--- Processing Complete ---")

    if all_volume_dice_scores:
        avg_volume_dice = np.mean(all_volume_dice_scores)
        avg_volume_jaccard = np.mean(all_volume_jaccard_scores)
        print(f"\nAverage Dice Score across {processed_volume_count} processed volumes: {avg_volume_dice:.4f}")
        print(f"Average Jaccard Score across {processed_volume_count} processed volumes: {avg_volume_jaccard:.4f}")
    else:
        print("No volumes were processed for metric calculation. Check your data paths and TotalSegmentator outputs.")