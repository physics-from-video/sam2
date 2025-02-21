import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.plotting import show_points, save_plot, overlay_mask_on_image
from skimage.measure import label, regionprops
import pickle

# Import SAM2 predictor builder (adjust the import path as needed)
from sam2.build_sam import build_sam2_video_predictor

# ----- CONFIGURATION -----
# Directory where the uniform processed videos are stored (from your previous processing).
# INPUT_VIDEOS_DIR = r"/scratch-shared/tnijdam/resized_generated_videos" # for generated videos
INPUT_VIDEOS_DIR = r"/scratch-shared/tnijdam/real-world-jpg" # for real-world videos

# Directory where the tracking results should be stored.
OUTPUT_DIR = r"/scratch-shared/tnijdam/sam2_tracking_centres/real_world"
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

def compute_mask_properties(mask, experiment):
    """
    Given a binary mask (2D numpy array), compute:
      - centre: the centroid of the mask (as (y, x))
      - max_distance: if experiment=="holonomic_pendulum", the maximum distance between any two mask pixels;
                      otherwise, None.
    """
    mask = np.squeeze(mask)  # Ensures shape is (H, W)
    mask_bool = mask > 0
    indices = np.argwhere(mask_bool)
    if indices.size == 0:
        centre = (None, None)
    else:
        centre = tuple(np.mean(indices, axis=0))
    
    max_distance = None
    if experiment == "holonomic_pendulum" and indices.size > 0:
        # Use regionprops to get major_axis_length as a proxy for max distance
        labeled = label(mask_bool)
        props = regionprops(labeled)
        if props:
            max_distance = props[0].major_axis_length
        else:
            max_distance = 0.0
    return centre, max_distance

def process_video_folder(video_folder):
    """
    For a given video folder (e.g. .../<experiment>/video_XX) that contains a subfolder
    "frames_for_tracking", run SAM2 tracking and produce:
      - A plot of 00000.jpg with the positive/negative labels overlaid.
      - A plot of the segmentation (mask) on the first frame.
      - Plots of segmentation every 5 frames.
    The outputs are saved inside OUTPUT_DIR with the same relative path as video_folder.
    """
    # Compute relative path from INPUT_VIDEOS_DIR
    rel_path = os.path.relpath(video_folder, INPUT_VIDEOS_DIR)
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
        save_plot(fig, os.path.join(tracking_plots_dir, f"clicks_object_{obj_id}.jpg"))
        

    
    # --- Initialize SAM2 tracking ---
    inference_state = predictor.init_state(video_path=frames_dir)
    predictor.reset_state(inference_state)
    # For each object, add its points separately.
    masks_to_show = {}
    
    for obj_id, data in object_points.items():
        # Add the points for this object; SAM2 should now track them separately.
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,  # Ensure correct object ID is used
            points=data["points"],
            labels=data["labels"],
        )
        
        masks_to_show[obj_id] = (out_mask_logits[obj_id - 1] > 0.0).cpu().numpy()

    for obj_id, mask in masks_to_show.items():
        print("Saving first frame segmentation overlay...")
        
        overlay_img = overlay_mask_on_image(first_frame, mask, obj_id=obj_id)
        save_path = os.path.join(tracking_plots_dir, f"segmentation_{obj_id}.jpg")
        
        Image.fromarray(overlay_img).save(save_path)
    

    # --- Plot segmentation on every frame ---
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        for i, out_obj_id in enumerate(out_obj_ids):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
    
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
                        key=lambda p: int(os.path.splitext(p)[0]))
    
    centers_over_time = {}   # {obj_id: [(frame_idx, (y, x)), ...]}
    masks_over_time = {}     # {frame_idx: {obj_id: mask}}
    max_distance_over_time = {}  # {obj_id: [(frame_idx, max_distance), ...]} (only if holonomic_pendulum)
    
    for out_frame_idx in range(0, len(frame_names)):
        frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        frame_img = np.array(Image.open(frame_path))

        if out_frame_idx in video_segments:
            masks_over_time[out_frame_idx] = {}

            for obj_id, mask in video_segments[out_frame_idx].items():
                frame_img = overlay_mask_on_image(frame_img, mask, obj_id=obj_id)

                # Compute the centre and max distance if applicable
                centre, max_distance = compute_mask_properties(mask, experiment)
                print(centre, max_distance)
                # Store mask
                masks_over_time[out_frame_idx][obj_id] = mask

                # Store centre over time
                if centre[0] is not None:
                    if obj_id not in centers_over_time:
                        centers_over_time[obj_id] = []
                    centers_over_time[obj_id].append((out_frame_idx, centre))

                # Store max distance only for "holonomic_pendulum"
                if experiment == "holonomic_pendulum":
                    if obj_id not in max_distance_over_time:
                        max_distance_over_time[obj_id] = []
                    max_distance_over_time[obj_id].append((out_frame_idx, max_distance))

        # Save the segmentation overlay image
        img_save_path = os.path.join(frames_plots, f"frame_{out_frame_idx:05d}_segmentation.jpg")
        Image.fromarray(frame_img).save(img_save_path)

    # --- Save all data at the end in a single pkl file per type ---
    centres_pkl_path = os.path.join(tracking_output_folder, "centres.pkl")
    masks_pkl_path = os.path.join(tracking_output_folder, "masks.pkl")

    with open(centres_pkl_path, "wb") as pkl_file:
        pickle.dump(centers_over_time, pkl_file)

    with open(masks_pkl_path, "wb") as pkl_file:
        pickle.dump(masks_over_time, pkl_file)

    if experiment == "holonomic_pendulum":
        max_distance_pkl_path = os.path.join(tracking_output_folder, "max_distance.pkl")
        with open(max_distance_pkl_path, "wb") as pkl_file:
            pickle.dump(max_distance_over_time, pkl_file)

    print(f"Saved centres to {centres_pkl_path}")
    print(f"Saved masks to {masks_pkl_path}")
    if experiment == "holonomic_pendulum":
        max_distance_pkl_path = os.path.join(tracking_output_folder, "max_distance.pkl")
        with open(max_distance_pkl_path, "wb") as pkl_file:
            pickle.dump(max_distance_over_time, pkl_file)
        print(f"Saved max distances to {max_distance_pkl_path}")
    print(f"Saved centres to {centres_pkl_path}")
    print(f"Saved masks to {masks_pkl_path}")
        
    if centers_over_time:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        for obj_id, points in centers_over_time.items():
            # Sort points by frame index
            points = sorted(points, key=lambda x: x[0])
            frame_indices = [pt[0] for pt in points]  # List of frame indices
            centres = np.array([pt[1] for pt in points], dtype=np.float32)  # Convert list of (y, x) tuples to array

            if centres.ndim != 2 or centres.shape[1] != 2:
                print(f"Warning: Invalid shape {centres.shape} for object {obj_id}, skipping.")
                continue  # Skip invalid data
            
            y_positions = centres[:, 0]  # Extract Y positions correctly
            x_positions = centres[:, 1]  # Extract X positions correctly

            # Plot X and Y positions over time
            ax1.plot(frame_indices, x_positions, marker='o', label=f"Object {obj_id} (X)")
            ax2.plot(frame_indices, y_positions, marker='o', label=f"Object {obj_id} (Y)")
        
        ax1.set_xlabel("Frame Index")
        ax1.set_ylabel("Centre X Position")
        ax1.legend()
        
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Centre Y Position")
        ax2.legend()
        
        plt.suptitle("Object Centres Over Time")
        
        centre_plot_path = os.path.join(tracking_output_folder, "centres_over_time.jpg")
        plt.savefig(centre_plot_path, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Centre trajectory plot saved to {centre_plot_path}")
    
    if experiment == "holonomic_pendulum" and max_distance_over_time:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for obj_id, distances in max_distance_over_time.items():
            # Sort distances by frame index
            distances = sorted(distances, key=lambda x: x[0])
            frame_indices = [d[0] for d in distances]  # Frame indices
            max_distances = np.array([d[1] for d in distances], dtype=np.float32)  # Extract max distances

            ax.plot(frame_indices, max_distances, marker='o', label=f"Object {obj_id}")

        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Max Distance")
        ax.legend()
        
        plt.suptitle("Max Distance Over Time (Holonomic Pendulum)")
        
        max_distance_plot_path = os.path.join(tracking_output_folder, "max_distance_over_time.jpg")
        plt.savefig(max_distance_plot_path, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Max distance trajectory plot saved to {max_distance_plot_path}")


    print(f"Tracking plots saved in {frames_plots}")


def process_all_videos_for_tracking():
    """
    Loops through all video folders in INPUT_VIDEOS_DIR (the uniform output from processing)
    that contain a "frames_for_tracking" folder, and runs tracking on each.
    """
    # for root, dirs, files in os.walk(INPUT_VIDEOS_DIR):
    #     # We assume video folders are named like "video_XX"
    #     if os.path.basename(root).startswith("video_"):
    #         # check if "frames_for_tracking" exists, other check if there are already jpgs files in this dir,because then we track this 
    #         frames_dir = os.path.join(root, "frames_for_tracking")
            
    #         if os.path.exists(frames_dir):
    #             print(f"Running tracking in {root}")
                
    #             # if "video_1" in root or "non_holonomic_pendulum" in root or "projectile" in root or "falling_ball" in root:
    #             #     print("----------------> SKIPPING")
    #             #     continue
    #             process_video_folder(root)

    
    # upload to HF
    OUTPUT_DIR = r"/scratch-shared/tnijdam/sam2_tracking_centres/real_world"
    repo_name = "physics-from-video/sam2-real-world-tracking"
    
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading {OUTPUT_DIR} to Hugging Face dataset repository {repo_name}...")

    # # Ensure the repository exists (create if necessary)
    api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)

    # # Delete old files from the repo before re-uploading to ensure overwrite
    api.delete_repo(repo_id=repo_name, repo_type="dataset")
    api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)

    # # Upload files (using upload_large_folder for large datasets)
    api.upload_large_folder(folder_path=OUTPUT_DIR, repo_id=repo_name, repo_type="dataset")

    print(f"Upload complete: {repo_name}")

if __name__ == "__main__":
    process_all_videos_for_tracking()
