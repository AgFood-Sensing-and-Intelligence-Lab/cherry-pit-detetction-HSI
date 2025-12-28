import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from spectral import open_image
from scipy.ndimage import binary_erosion


MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"

IMG_FOLDER = " "        # folder containing .img and .hdr
IMG_NAME = "cherry_cube.img"            # hyperspectral cube filename
BAND_TO_USE = 1014                      # band index with best contrats


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.56,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=200
)


cube_path = os.path.join(IMG_FOLDER, IMG_NAME)
cube = open_image(cube_path).load()       # shape: (H, W, Bands)
print(f"Cube shape: {cube.shape}")

# Extract band 1014 (grayscale)
band = cube[:, :, BAND_TO_USE]

# Convert to 3-channel (so SAM can work)
band_rgb = cv2.cvtColor((band / band.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


# Segment Using SAM ---

sam_result = mask_generator.generate(band_rgb)

# Filter masks by reasonable area
filtered_masks = [
    m for m in sam_result
    if 1500 <= np.sum(m['segmentation']) <= 8000]

mask = filtered_masks[0]['segmentation'].astype(np.uint8) # take first valid mask

mask = binary_erosion(mask, structure=np.ones((5,5)), iterations=3).astype(np.uint8)

# Propagate Mask to Hypercube
masked_pixels = cube[mask == 1, :] 

#Extract Average Spectrum ---

avg_spectrum = masked_pixels.mean(axis=0)
np.save('avg_spectrum.npy', avg_spectrum)

