import matplotlib.pyplot as plt
import numpy as np

# def show_mask(mask, ax, obj_id=None):
#     """Overlay segmentation masks with distinct colors per object."""
#     cmap = plt.get_cmap("tab10")  # Use tab10 colormap for distinct objects
#     cmap_idx = obj_id % 10 if obj_id is not None else 0  # Ensure cycling through colors
#     color = np.array([*cmap(cmap_idx)[:3], 0.6])  # Extract RGB + alpha transparency

#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
    
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id + 1
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
# def show_mask(mask, ax, obj_id=None):
#     """Overlay segmentation masks with two highly distinctive colors."""
#     # Define two strong, easily distinguishable colors (e.g., red and blue)
#     colors = [
#         np.array([1.0, 0.0, 0.0, 0.6]),  # Red (RGBA)
#         np.array([0.0, 0.0, 1.0, 0.6])   # Blue (RGBA)
#     ]

#     # Default to first color if obj_id is None, cycle between two colors
#     color = colors[obj_id % 2] if obj_id is not None else colors[0]

#     # Apply mask with the selected color
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def show_mask(mask, ax, obj_id=None):
#     """
#     Overlay a segmentation mask on an axis with a distinctive color:
#       - If obj_id is 0 (or None), use red (RGBA: [1, 0, 0, 0.5]).
#       - If obj_id is 1, use blue (RGBA: [0, 0, 1, 0.5]).
    
#     With only two objects, overlapping areas will blend to a purple hue.
#     """
#     # Ensure the mask is 2D (if it has extra channels, take the first one)
#     if mask.ndim > 2:
#         mask = mask[..., 0]
#     h, w = mask.shape
#     # Choose the color based on object id:
#     if obj_id == 1:
#         color = np.array([0, 0, 1, 0.5])  # Blue, 50% opacity.
#     else:
#         color = np.array([1, 0, 0, 0.5])  # Red, 50% opacity.
#     # Create an overlay image: for pixels where mask > 0, set the color.
#     overlay = np.zeros((h, w, 4), dtype=np.float32)
#     overlay[mask > 0] = color
#     ax.imshow(overlay)

    
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
