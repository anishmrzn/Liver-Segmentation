import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import io, transform, morphology, measure
from skimage.exposure import equalize_hist # Added for histogram equalization if needed for visualization, not core preprocessing
import nibabel as nib
from tqdm import tqdm

# Import necessary MONAI components for UNet inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd, # Added based on liv_seg.ipynb
    Spacingd,     # Added based on liv_seg.ipynb
    SpatialPadd,  # Added based on liv_seg.ipynb
    Resized,      # This will be for UNet's input
    ToTensord,
    AsDiscrete, # For post-processing UNet output
)
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference


# --- Configuration Paths ---
TEST_VOLUMES_PATH = 'D:/monai-project/Data_Train_Test/TestVolumes'
TEST_SEGMENTATION_PATH = 'D:/monai-project/Data_Train_Test/TestSegmentation'
OUTPUT_PLOTS_PATH = 'D:/monai-project/segmentation_plots_unet_medsam_volume_metrics' # New output path for UNet-MedSAM plots
# --- IMPORTANT CHANGE: Update UNET_MODEL_PATH ---
UNET_MODEL_PATH = 'D:/monai-project/asa/liver_seg_unet_2class_patch.pth' # Path to your trained UNet model from liv_seg.ipynb

MEDSAM_CKPT_PATH = 'medsam_vit_b.pth'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create output directories
os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)
print(f"Segmentation plots will be saved to: {OUTPUT_PLOTS_PATH}")

MIN_COMPONENT_AREA = 500
MORPH_OPEN_DISK = 2
MORPH_CLOSE_DISK = 5

# --- Helper functions (from your totalseg_medsam.py and liv_seg.ipynb) ---

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
    # --- FIX START ---
    # The first argument to plt.Rectangle should be a tuple (x, y)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=lw))
    # --- FIX END ---

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
    # This normalization is for visualization purposes only, not for UNet input
    min_val = np.min(image_slice)
    max_val = np.max(image_slice)
    if max_val - min_val == 0:
        return np.zeros_like(image_slice, dtype=np.uint8)
    normalized_slice = 255 * (image_slice - min_val) / (max_val - min_val)
    return normalized_slice.astype(np.uint8)

if __name__ == '__main__':
    # --- Load your trained UNet model ---
    print("Loading trained UNet model...")
    unet_model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2, # Assuming your UNet outputs 2 channels (background + foreground)
        # --- IMPORTANT CHANGE: Update UNet channels based on liv_seg.ipynb ---
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        # --- NOTE: liv_seg.ipynb doesn't explicitly specify 'norm', keeping default here ---
        # norm=Norm.BATCH, # If your trained model used this, uncomment
    ).to(device)

    # Load the best performing weights
    if os.path.exists(UNET_MODEL_PATH):
        unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device)) # Add map_location for consistency
        print(f"UNet model loaded successfully from {UNET_MODEL_PATH}.")
    else:
        print(f"Error: UNet model checkpoint not found at {UNET_MODEL_PATH}. Please train your UNet model first or verify the path.")
        exit() # Exit if UNet model is not found

    unet_model.eval() # Set UNet to evaluation mode

    # --- Define UNet preprocessing transforms (MATCHING liv_seg.ipynb) ---
    # These transforms must precisely match the 'preprocess' pipeline in liv_seg.ipynb
    # for inference to work correctly with the trained model.
    # The `crop_size` was (128, 128, 64) in liv_seg.ipynb
    crop_size = (128, 128, 64)

    unet_preprocessing_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            EnsureChannelFirstD(keys=["vol"]),
            Orientationd(keys=["vol"], axcodes="RAS"), # From liv_seg.ipynb
            Spacingd(keys=["vol"], pixdim=(1,1,3), mode=("bilinear")), # From liv_seg.ipynb, mode for image
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            # SpatialPadd ensures the volume is at least crop_size. Sliding window inference handles larger.
            SpatialPadd(keys=["vol"], spatial_size=crop_size), # From liv_seg.ipynb
            ToTensord(keys=["vol"]),
        ]
    )


    # --- Load MedSAM model ---
    print("Loading MedSAM model...")
    if not os.path.exists(MEDSAM_CKPT_PATH):
        import urllib.request
        model_url = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
        print(f"Downloading MedSAM model from {model_url} to {MEDSAM_CKPT_PATH}...")
        try:
            urllib.request.urlretrieve(model_url, MEDSAM_CKPT_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download MedSAM model: {e}. Please download it manually if the link is broken or network issues persist.")
            exit()


    medsam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    print("MedSAM model loaded successfully.")

    all_volume_dice_scores = []
    all_volume_jaccard_scores = []
    processed_volume_count = 0

    volume_files = sorted([f for f in os.listdir(TEST_VOLUMES_PATH) if f.endswith('.nii.gz')])

    print("\n--- Starting UNet Inference and MedSAM Integration ---")

    with torch.no_grad(): # Ensure no gradient calculations for inference
        for vol_fname in tqdm(volume_files, desc="Processing Volumes (UNet + MedSAM)"):
            vol_path = os.path.join(TEST_VOLUMES_PATH, vol_fname)
            gt_seg_fname = vol_fname.replace('volume', 'segmentation').replace('img', 'label')
            if not gt_seg_fname.endswith('.nii.gz'):
                gt_seg_fname = os.path.splitext(gt_seg_fname)[0] + '.nii.gz'
            gt_seg_path = os.path.join(TEST_SEGMENTATION_PATH, gt_seg_fname)

            if not os.path.exists(gt_seg_path):
                print(f"Warning: Corresponding ground truth segmentation not found for {vol_fname}. Skipping.")
                continue

            try:
                # Load the original NIfTI volume and ground truth for evaluation and visualization
                nifti_vol_original = nib.load(vol_path) # Load as nibabel object to preserve affine/header
                nifti_gt_seg_original = nib.load(gt_seg_path) # Load as nibabel object


                # --- UNet Inference Step ---
                # Apply UNet's specific preprocessing to the current volume
                # The keys will be 'vol' after this transform
                data_for_unet_transform = unet_preprocessing_transforms({"vol": vol_path})
                input_tensor_for_unet = data_for_unet_transform["vol"].unsqueeze(0).to(device) # Add batch dim


                # Perform UNet inference using sliding window on the preprocessed volume
                # roi_size should match the trained patch size
                unet_output = sliding_window_inference(
                    inputs=input_tensor_for_unet,
                    roi_size=crop_size, # Use the crop_size from training
                    sw_batch_size=1,
                    predictor=unet_model,
                    overlap=0.5, # Standard overlap for sliding window, can be tuned
                )
                # Apply argmax to get the class prediction (e.g., liver)
                unet_pred_monai_res = torch.argmax(unet_output, dim=1).cpu().numpy().squeeze(0) # Output is [H, W, D] at MONAI processed resolution


                # Now, resize the UNet prediction volume back to the original NIfTI volume's spatial dimensions.
                # This must be done carefully to match spacing and orientation.
                # It's generally better to apply the inverse transforms or resample.
                # For simplicity here, we'll resize to the original data shape, but be aware of potential resampling artifacts.

                # Determine the target shape based on the original NIfTI image's data
                original_shape_spatial = nifti_vol_original.shape
                # If your original image has different spacing than (1,1,3), simply resizing might not be ideal.
                # It's crucial that `Spacingd` in `unet_preprocessing_transforms` makes the image isotropic.

                # Get the processed image affine and header from MONAI's LoadImaged + Spacingd
                # This requires more complex handling with Monai's SaveImaged or custom resampling.
                # For a quick fix, we assume direct resize after argmax is sufficient IF spacing makes volumes comparable.

                # Alternative: Use MONAI's InverseTransform or separate Resample
                # For now, let's stick to skimage.transform.resize for simplicity as it worked for the previous fix,
                # but with an important note about `order=0` for masks.
                unet_pred_full_res = transform.resize(
                    unet_pred_monai_res.astype(np.float32),
                    original_shape_spatial,
                    order=0, # Use nearest neighbor for masks to keep them binary
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.uint8)

                # Get ground truth data and ensure it's also aligned to original space if needed
                nifti_gt_seg = nifti_gt_seg_original.get_fdata().astype(np.uint8)


                num_slices = nifti_vol_original.shape[2]
                print(f"\nProcessing {vol_fname} with {num_slices} slices using UNet and MedSAM...")

                medsam_seg_volume = np.zeros_like(nifti_gt_seg, dtype=np.uint8)
                unet_liver_mask_volume = unet_pred_full_res # Already resized to original shape

                # Find a representative slice for plotting (where ground truth liver exists)
                representative_slice_idx = -1
                gt_liver_slices = np.where(np.any(nifti_gt_seg == 1, axis=(0, 1)))[0]
                if len(gt_liver_slices) > 0:
                    # Pick a slice around the middle of the liver region
                    representative_slice_idx = int(np.median(gt_liver_slices))
                else:
                    # Fallback to middle slice if no liver in GT
                    representative_slice_idx = num_slices // 2

                if representative_slice_idx == -1: # Should not happen with fallback, but good to check
                    print(f"Warning: No ground truth liver found in {vol_fname}. Cannot select representative slice for plot. Skipping plots.")
                    # Setting to an invalid index to skip plotting logic later
                    representative_slice_idx = -2


                # Iterate through slices for MedSAM inference
                for s_idx in range(num_slices):
                    image_slice_hu = nifti_vol_original.get_fdata()[:, :, s_idx] # Get original slice
                    gt_mask_slice = nifti_gt_seg[:, :, s_idx].astype(np.uint8)

                    # Get UNet's prediction for the current slice (already full-res)
                    unet_slice_for_prompt = unet_liver_mask_volume[:, :, s_idx]


                    # Generate prompt box from UNet's segmentation slice
                    prompt_mask, prompt_box = get_bbox_from_mask(
                        unet_slice_for_prompt, MIN_COMPONENT_AREA, padding_percentage=0.10
                    )

                    medsam_seg_slice = np.zeros_like(gt_mask_slice, dtype=np.uint8)
                    if prompt_box is None:
                        # If UNet didn't find anything, MedSAM won't have a prompt
                        pass
                    else:
                        img_display = normalize_image(image_slice_hu)
                        H, W = img_display.shape[:2] # Original dimensions

                        img_3c = np.repeat(img_display[:, :, None], 3, axis=-1)
                        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
                        # Normalize MedSAM input to [0,1]
                        img_1024 = (img_1024 - img_1024.min()) / np.clip(
                            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                        )
                        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

                        image_embedding = medsam_model.image_encoder(img_1024_tensor)

                        box_np_for_medsam = np.array([prompt_box])
                        box_1024_for_medsam = box_np_for_medsam / np.array([W, H, W, H]) * 1024

                        medsam_seg_slice = medsam_inference(medsam_model, image_embedding, box_1024_for_medsam, H, W)

                    medsam_seg_volume[:, :, s_idx] = medsam_seg_slice

                    if s_idx == representative_slice_idx:
                        img_display_for_plot = normalize_image(image_slice_hu)
                        current_dice_medsam = dice_score(medsam_seg_slice, gt_mask_slice)
                        current_jaccard_medsam = jaccard_score(medsam_seg_slice, gt_mask_slice)
                        current_dice_unet = dice_score(unet_slice_for_prompt, gt_mask_slice) # Dice for UNet's slice
                        current_jaccard_unet = jaccard_score(unet_slice_for_prompt, gt_mask_slice) # Jaccard for UNet's slice

                        fig_seg, ax_seg = plt.subplots(1, 5, figsize=(30, 6)) # 5 subplots now
                        fig_seg.suptitle(f"{os.path.splitext(vol_fname)[0]} - Slice {s_idx} (Rep. Plot) | UNet Dice: {current_dice_unet:.4f}, MedSAM Dice: {current_dice_medsam:.4f}", fontsize=12)

                        ax_seg[0].imshow(img_display_for_plot, cmap='gray')
                        if prompt_box is not None:
                            show_box(prompt_box, ax_seg[0], color='red')
                        ax_seg[0].set_title('Original Image + UNet-Derived Prompt Box')
                        ax_seg[0].axis('off')

                        ax_seg[1].imshow(img_display_for_plot, cmap='gray')
                        show_mask(unet_slice_for_prompt, ax_seg[1], random_color=False)
                        ax_seg[1].set_title('UNet Liver Mask (Prompt Source)')
                        ax_seg[1].axis('off')

                        ax_seg[2].imshow(img_display_for_plot, cmap='gray')
                        show_mask(gt_mask_slice, ax_seg[2], random_color=False)
                        ax_seg[2].set_title('Ground Truth Liver')
                        ax_seg[2].axis('off')

                        ax_seg[3].imshow(img_display_for_plot, cmap='gray')
                        show_mask(medsam_seg_slice, ax_seg[3], random_color=False)
                        ax_seg[3].set_title('MedSAM Segmentation (from UNet Prompt)')
                        ax_seg[3].axis('off')

                        # Overlay for comparison
                        ax_seg[4].imshow(img_display_for_plot, cmap='gray')
                        show_mask(gt_mask_slice, ax_seg[4], random_color=True) # GT in one color
                        show_mask(medsam_seg_slice, ax_seg[4], random_color=True) # MedSAM in another
                        ax_seg[4].set_title('GT (Greenish) vs. MedSAM (Yellowish)')
                        ax_seg[4].axis('off')


                        plt.tight_layout(rect=[0, 0.03, 1, 0.9])

                        plot_filename = f"{os.path.splitext(vol_fname)[0].replace('.nii','')}_slice_{s_idx:04d}_unet_medsam_rep_plot.png"
                        plot_filepath = os.path.join(OUTPUT_PLOTS_PATH, plot_filename)
                        plt.savefig(plot_filepath, bbox_inches='tight', dpi=100)
                        plt.close() # Always close the plot to free memory


                # Calculate volume-wise Dice and Jaccard for MedSAM and UNet
                volume_dice_medsam = dice_score(medsam_seg_volume, nifti_gt_seg)
                volume_jaccard_medsam = jaccard_score(medsam_seg_volume, nifti_gt_seg)

                volume_dice_unet = dice_score(unet_liver_mask_volume, nifti_gt_seg)
                volume_jaccard_unet = jaccard_score(unet_liver_mask_volume, nifti_gt_seg)


                all_volume_dice_scores.append({"unet": volume_dice_unet, "medsam": volume_dice_medsam})
                all_volume_jaccard_scores.append({"unet": volume_jaccard_unet, "medsam": volume_jaccard_medsam})

                processed_volume_count += 1

                print(f"   Volume {os.path.splitext(os.path.splitext(vol_fname)[0])[0]}: UNet Dice = {volume_dice_unet:.4f}, UNet Jaccard = {volume_jaccard_unet:.4f} | MedSAM Dice = {volume_dice_medsam:.4f}, MedSAM Jaccard = {volume_jaccard_medsam:.4f}")

            except Exception as e:
                print(f"Error processing {vol_fname}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n--- Processing Complete ---")

    if all_volume_dice_scores:
        avg_volume_dice_unet = np.mean([d["unet"] for d in all_volume_dice_scores])
        avg_volume_jaccard_unet = np.mean([j["unet"] for j in all_volume_jaccard_scores])
        avg_volume_dice_medsam = np.mean([d["medsam"] for d in all_volume_dice_scores])
        avg_volume_jaccard_medsam = np.mean([j["medsam"] for j in all_volume_jaccard_scores])

        print(f"\nAverage UNet Dice Score across {processed_volume_count} processed volumes: {avg_volume_dice_unet:.4f}")
        print(f"Average UNet Jaccard Score across {processed_volume_count} processed volumes: {avg_volume_jaccard_unet:.4f}")
        print(f"Average MedSAM Dice Score (UNet Prompt) across {processed_volume_count} processed volumes: {avg_volume_dice_medsam:.4f}")
        print(f"Average MedSAM Jaccard Score (UNet Prompt) across {processed_volume_count} processed volumes: {avg_volume_jaccard_medsam:.4f}")

        # Plotting the overall average scores
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(2)

        # Separate bars for UNet and MedSAM
        plt.bar(index, [avg_volume_dice_unet, avg_volume_jaccard_unet], bar_width, label='UNet (Initial Seg)', color='skyblue')
        plt.bar(index + bar_width, [avg_volume_dice_medsam, avg_volume_jaccard_medsam], bar_width, label='MedSAM (UNet Prompted)', color='lightcoral')

        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Average Liver Segmentation Scores (UNet vs. UNet-Prompted MedSAM)')
        plt.xticks(index + bar_width / 2, ['Dice Score', 'Jaccard Score'])
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_summary_filepath = os.path.join(OUTPUT_PLOTS_PATH, "summary_scores_unet_medsam.png")
        plt.savefig(plot_summary_filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Summary plot saved to: {plot_summary_filepath}")


    else:
        print("No volumes were processed for metric calculation. Check your data paths and UNet model output.")

