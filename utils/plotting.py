import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def overlay_mask_on_image(image, mask, obj_id=None, random_color=False):
    """Overlay segmentation mask on the original image without using matplotlib."""
    if random_color:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id + 1
        color = (np.array(cmap(cmap_idx)[:3]) * 255).astype(np.uint8)

    # Ensure mask is in (height, width) format
    mask = np.squeeze(mask)  # Remove singleton dimensions (e.g., (1, H, W) -> (H, W))
    
    if mask.shape[:2] != image.shape[:2]:  # Ensure mask and image match in size
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")

    # Convert grayscale mask to a 3-channel color mask
    mask_rgb = np.zeros_like(image, dtype=np.uint8)
    mask_rgb[mask > 0] = color  # Apply color to masked regions
    
    # Blend original image and mask overlay
    alpha = 0.6  # Transparency factor
    blended = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

    return blended



# def overlay_mask_on_image(image, mask, obj_id=None, random_color=False):
#     """Overlay segmentation mask on the original image without using matplotlib."""
#     if random_color:
#         color = np.random.randint(0, 255, (3,), dtype=np.uint8)
#     else:
#         cmap = plt.get_cmap("tab10")
#         cmap_idx = 0 if obj_id is None else obj_id + 1
#         color = (np.array(cmap(cmap_idx)[:3]) * 255).astype(np.uint8)

#     # Convert mask to three-channel color
#     mask_rgb = np.zeros_like(image, dtype=np.uint8)
#     mask_rgb[mask > 0] = color  # Apply color to masked regions
    
#     # Blend original image and mask overlay
#     alpha = 0.6  # Transparency factor
#     blended = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

#     return blended


def show_points(coords, labels, ax, marker_size=200):
    """Plot positive (green) and negative (red) points on an axis."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if pos_points.size > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
    if neg_points.size > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

def save_plot(fig, save_path):
    """Save a matplotlib figure and close it."""
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
