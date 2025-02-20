import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.plotting import show_mask, show_points, save_plot

# Import SAM2 predictor builder (adjust the import path as needed)
from sam2.build_sam import build_sam2_video_predictor

# ----- CONFIGURATION -----
# Directory where the uniform processed videos are stored (from your previous processing).
RESIZED_VIDEOS_DIR = r"/scratch-shared/tnijdam/resized_generated_videos"
# Directory where the tracking results should be stored.
OUTPUT_DIR = r"/scratch-shared/tnijdam/sam2_tracking_centres"
# Path to the JSON file with labels.
LABELS_JSON_PATH = r"/home/tnijdam/VGMs/prompts/reference_images/labels.json"

# SAM2 model configuration
CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"

# Build the SAM2 video predictor.
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)

# Mapping function: for holonomic_pendulum, key is "video_5"; otherwise, "video_0".
def get_video_key(experiment):
    return "video_5" if experiment == "holonomic_pendulum" else "video_0"

# Load the labels JSON.
with open(LABELS_JSON_PATH, "r") as f:
    labels_json = json.load(f)

def process_video_folder(video_folder):
    """
    For a given video folder (e.g. .../<experiment>/video_XX) that contains a subfolder
    "frames_for_tracking", run SAM2 tracking and produce:
      - A plot of 00000.jpg with the positive/negative labels overlaid.
      - A plot of the segmentation (mask) on the first frame.
      - Plots of segmentation every 5 frames.
    The outputs are saved inside OUTPUT_DIR with the same relative path as video_folder.
    """
    # Compute relative path from RESIZED_VIDEOS_DIR
    rel_path = os.path.relpath(video_folder, RESIZED_VIDEOS_DIR)
    # Compute new output path inside OUTPUT_DIR
    tracking_output_folder = os.path.join(OUTPUT_DIR, rel_path)
    frames_dir = os.path.join(video_folder, "frames_for_tracking")
    os.makedirs(tracking_output_folder, exist_ok=True)
    
    parts = rel_path.split(os.sep)
    if len(parts) < 2:
        print(f"Unexpected folder structure: {video_folder}")
        return
    experiment = parts[-2]
    video_key = get_video_key(experiment)
    
    # Look up label data for this experiment and video.
    if experiment not in labels_json:
        print(f"No label data for experiment {experiment}")
        return
    if video_key not in labels_json[experiment]:
        print(f"No label data for video key {video_key} in experiment {experiment}")
        return
    label_info = labels_json[experiment][video_key]
    
    # Process points for each object separately.
    object_points = {}
    for obj_key, obj_data in label_info.items():
        obj_id = int(obj_key.split("_")[-1])  # e.g., "object_1" -> 1
        pos = np.array(obj_data.get("positive", []), dtype=np.float32)
        neg = np.array(obj_data.get("negative", []), dtype=np.float32)
        pts = []
        labs = []
        if pos.size > 0:
            pts.append(pos)
            labs.append(np.ones(len(pos), dtype=int))
        if neg.size > 0:
            pts.append(neg)
            labs.append(np.zeros(len(neg), dtype=int))
        if pts:
            object_points[obj_id] = {
                "points": np.concatenate(pts, axis=0),
                "labels": np.concatenate(labs, axis=0)
            }
    
    # For plotting the points on the first frame, combine points from all objects.
    if not object_points:
        print(f"No labeled points for experiment {experiment}, video {video_key}")
        return
    
    # Ensure frames exist.
    if not os.path.exists(frames_dir):
        print(f"Frames folder not found in {video_folder}")
        return
    
    # Create tracking output directory inside OUTPUT_DIR.
    tracking_plots_dir = os.path.join(tracking_output_folder, "tracking_plots")
    frames_plots = os.path.join(tracking_plots_dir, "frames")
    os.makedirs(frames_plots, exist_ok=True)
    
    # --- Plot 00000.jpg with overlaid labels (combined for visualization) ---
    first_frame_path = os.path.join(frames_dir, "00000.jpg")
    if not os.path.exists(first_frame_path):
        print(f"First frame not found: {first_frame_path}")
        return
    
    first_frame = np.array(Image.open(first_frame_path))
    for obj_id, data in object_points.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"First Frame Labels for Object {obj_id}")
        ax.imshow(first_frame)
        # Plot only the points and labels for this object.
        show_points(data["points"], data["labels"], ax)
        save_plot(fig, os.path.join(tracking_plots_dir, f"first_frame_labels_object_{obj_id}.jpg"))
    # --- Initialize SAM2 tracking ---
    inference_state = predictor.init_state(video_path=frames_dir)
    predictor.reset_state(inference_state)
    # For each object, add its points separately.
    masks_to_show = {}
    
    for obj_id, data in object_points.items():
        print("HEEEEEEEEEEEREEEE1111")
        print(obj_id, data)
        # Add the points for this object; SAM2 should now track them separately.
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,  # Ensure correct object ID is used
            points=data["points"],
            labels=data["labels"],
        )
        
        masks_to_show[obj_id] = (out_mask_logits[obj_id - 1] > 0.0).cpu().numpy()

    # --- Plot segmentation on first frame ---
    # Here we assume that out_mask_logits from the last call corresponds to each object.
    # For simplicity, we use the mask of the first object.

    # Loop through all stored masks and overlay them using show_mask.
    for obj_id, mask in masks_to_show.items():
        print("HEEEEEEEEEEEREEEE222")
        print(obj_id, mask)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title("First Frame Segmentation (All Objects)")
        ax.imshow(first_frame)
        show_mask(mask, ax, obj_id=obj_id)
        save_plot(fig, os.path.join(tracking_plots_dir, f"first_frame_segmentation_{obj_id}.jpg"))
    
    # --- Propagate segmentation over the video ---
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        for i, out_obj_id in enumerate(out_obj_ids):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
    
    # --- Plot segmentation every 5 frames ---
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
                         key=lambda p: int(os.path.splitext(p)[0]))
    for out_frame_idx in range(0, len(frame_names), 5):
        frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        frame_img = np.array(Image.open(frame_path))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Frame {out_frame_idx} Segmentation")
        ax.imshow(frame_img)
        if out_frame_idx in video_segments:
            for obj_id, mask in video_segments[out_frame_idx].items():
                show_mask(mask, ax, obj_id=obj_id)
        save_plot(fig, os.path.join(frames_plots, f"frame_{out_frame_idx:05d}_segmentation.jpg"))
    
    print(f"Tracking plots saved in {frames_plots}")


def process_all_videos_for_tracking():
    """
    Loops through all video folders in RESIZED_VIDEOS_DIR (the uniform output from processing)
    that contain a "frames_for_tracking" folder, and runs tracking on each.
    """
    for root, dirs, files in os.walk(RESIZED_VIDEOS_DIR):
        # We assume video folders are named like "video_XX"
        if os.path.basename(root).startswith("video_"):
            frames_dir = os.path.join(root, "frames_for_tracking")
            if os.path.exists(frames_dir):
                print(f"Running tracking in {root}")
                
                if "video_1" in root or "non_holonomic_pendulum" in root or "projectile" in root or "falling_ball" in root:
                    print("----------------> SKIPPING")
                    continue
                process_video_folder(root)

if __name__ == "__main__":
    process_all_videos_for_tracking()
