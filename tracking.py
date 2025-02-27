import os
import numpy as np
from PIL import Image
from utils.processing import load_labels, process_object_points, process_segmentation_frames, save_tracking_data
from utils.plotting import save_first_frame_segmentation, plot_first_frame, plot_centres_over_time, plot_max_distance_over_time
from sam2.build_sam import build_sam2_video_predictor
import argparse
import re
from transformers import pipeline
import torch
from torchvision import transforms

# ----- CONFIGURATION -----
LABELS_JSON_PATH = r"../../prompts/reference_images/labels.json"
CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"

# Build the SAM2 video predictor.
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)

class DepthProcessor:
    def __init__(self, encoder='vitl'):
        self.mapper = {"vits": "small", "vitb": "base", "vitl": "large"}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_anything = pipeline(
            task="depth-estimation", 
            model=f"nielsr/depth-anything-{self.mapper[encoder]}", 
            device=0 if self.device == 'cuda' else -1
        )
        self.to_tensor = transforms.ToTensor()
    
    def predict_depth(self, image):
        depth_output = self.depth_anything(image)
        return self.to_tensor(depth_output["depth"]).squeeze().to(self.device)

def process_depth_frames(frames_dir, masks_over_time):
    """
    Process depth for each frame using DepthAnything and compute average depth for each object's mask.
    Returns depth_over_time dictionary mapping object IDs to frame-wise depth values.
    """
    depth_processor = DepthProcessor()
    depth_over_time = {}

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])
    )
    max_frame_idx = len(frame_files)
    objects_names = masks_over_time[0].keys()
    depth_over_time = {obj_name: np.nan * np.ones(max_frame_idx) for obj_name in objects_names}
    for frame_idx, frame_file in enumerate(frame_files):
        assert f"{frame_idx:05d}.jpg" == frame_file
        frame_path = os.path.join(frames_dir, frame_file)
        image = Image.open(frame_path).convert('RGB')
        depth = depth_processor.predict_depth(image)
        
        
        if frame_idx in masks_over_time:
            for obj_id, mask in masks_over_time[frame_idx].items():
                mask_tensor = torch.from_numpy(mask).to(depth.device)
                masked_depth = depth * mask_tensor
                avg_depth = masked_depth.sum() / (mask_tensor.sum() + 1e-6)
                depth_over_time[obj_id][frame_idx] = avg_depth.item()
    return depth_over_time

def initialize_tracking(frames_dir, object_points):
    """
    Initialize the SAM2 tracking with the provided object points.
    Returns the inference state and initial masks.
    """
    inference_state = predictor.init_state(video_path=frames_dir)
    predictor.reset_state(inference_state)
    masks_to_show = {}
    for obj_id, data in object_points.items():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=data["points"],
            labels=data["labels"],
        )
        masks_to_show[obj_id] = (out_mask_logits[obj_id - 1] > 0.0).cpu().numpy()
    return inference_state, masks_to_show


def propagate_tracking(inference_state):
    """Run the segmentation propagation over the video and collect masks."""
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        for i, out_obj_id in enumerate(out_obj_ids):
            video_segments.setdefault(out_frame_idx, {})[out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments


def process_video_folder(video_folder, labels_json, args):
    """
    Process a single video folder:
      - Check for caching.
      - Prepare labels and output directories.
      - Process first frame.
      - Initialize tracking and save initial segmentation.
      - Propagate segmentation and process every 10th frame.
      - Save data and plot trajectories.
    """
    rel_path = os.path.relpath(video_folder, args.input_dir)
    tracking_output_folder = os.path.join(args.output_dir, rel_path)
    if args.use_cache and os.path.exists(tracking_output_folder) and os.listdir(tracking_output_folder):
        print(f"Skipping cached folder: {tracking_output_folder}")
        return
    os.makedirs(tracking_output_folder, exist_ok=True)

    frames_dir = os.path.join(video_folder, "frames_for_tracking")
    if not os.path.exists(frames_dir):
        print(f"Frames folder not found in {video_folder}")
        return

    parts = rel_path.split(os.sep)
    if len(parts) < 2:
        print(f"Unexpected folder structure: {video_folder}")
        return

    experiment = parts[-2]
    
    # we need to rename the experiment to match the labels
    if experiment == "non_holonomic_pendulum":
        experiment = "non_holonomic"
    
    # Extract the last part of the path, which contains the video identifier
    video_key_str = parts[-1]

    # Determine the key used for extracting labels:
    # - For real-world videos, the key is derived from the video filename.
    # - For generated videos, we use a fixed key to ensure consistency in label extraction.
    if args.real_world:
        video_number = int(video_key_str.split("_")[1])
        video_key = f"video_{video_number}"
    else:
        # Use a predefined key for generated videos based on the experiment type
        video_key = "video_5" if experiment == "holonomic_pendulum" else "video_0"

    if experiment not in labels_json or video_key not in labels_json[experiment]:
        print(f"Label info missing for experiment '{experiment}' and video '{video_key}'")
        return

    label_info = labels_json[experiment][video_key]
    object_points = process_object_points(label_info)
    if not object_points:
        print(f"No labeled points for video {video_key}")
        return

    # Prepare output directories for plots.
    tracking_plots_dir = os.path.join(tracking_output_folder, "tracking_plots")
    os.makedirs(os.path.join(tracking_plots_dir, "frames"), exist_ok=True)

    # Process and plot the first frame.
    first_frame_path = os.path.join(frames_dir, "00000.jpg")
    if not os.path.exists(first_frame_path):
        print(f"First frame not found: {first_frame_path}")
        return
    first_frame = np.array(Image.open(first_frame_path))
    plot_first_frame(first_frame, object_points, tracking_plots_dir)

    # Initialize SAM2 tracking and save initial segmentation overlay.
    inference_state, masks_to_show = initialize_tracking(frames_dir, object_points)
    save_first_frame_segmentation(first_frame, masks_to_show, tracking_plots_dir)

    # Propagate tracking over the video.
    video_segments = propagate_tracking(inference_state)
    centers_over_time, masks_over_time, max_distance_over_time = process_segmentation_frames(
        frames_dir, video_segments, tracking_plots_dir, experiment
    )
    depth_over_time = process_depth_frames(frames_dir, masks_over_time)

    centers3d_over_time = {}
    for key in centers_over_time.keys():
        txy = centers_over_time[key]
        z_over_time = depth_over_time[key]
        xy = np.array([xy for _, xy in txy])
        t = np.array([t for t, _ in txy])
        txy = np.concatenate([t[:, None], xy, z_over_time[:, None]], axis=1)
        centers3d_over_time[key] = txy
    
    
    # Save computed tracking data.
    save_tracking_data(tracking_output_folder, centers_over_time, centers3d_over_time, masks_over_time, max_distance_over_time, experiment)
    # Plot trajectories.
    plot_centres_over_time(centers_over_time, tracking_output_folder)
    if experiment == "holonomic_pendulum":
        plot_max_distance_over_time(max_distance_over_time, tracking_output_folder)

    print(f"Finished processing video: {video_folder}")


def process_all_videos(args):
    """Loop through all video folders and process those containing a 'frames_for_tracking' directory."""
    labels_json = load_labels(LABELS_JSON_PATH)
    
    # sort
    
    for root, dirs, files in os.walk(args.input_dir):
        # Extract the base directory name
        base_name = os.path.basename(root)

        # Check the condition
        # Real world videos are called video_0, video_1, etc.
        # Generated videos are called 0, 1, etc.
        if (args.real_world and base_name.startswith("video_")) or (not args.real_world and re.match(r"^\d{1,2}$", base_name)):
            frames_dir = os.path.join(root, "frames_for_tracking")
            # we dont have the labels for this yet so skipping it for now
            if any(skip in root for skip in []): #["double_pendulum", "holonomic_pendulum", "falling_ball", "projectile"]): #  "non_holonomic"
                print(f"Skipping folder: {root}")
                continue
            if os.path.exists(frames_dir):
                print(f"Processing folder: {root}")
                process_video_folder(root, labels_json, args)

    if args.upload_to_huggingface:
        # Upload results to Hugging Face.
        from huggingface_hub import HfApi
        api = HfApi()
        repo_name = "physics-from-video/sam2-tracking"
        print(f"Uploading {args.output_dir} to Hugging Face dataset repository {repo_name}...")
        api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(folder_path=args.output_dir, repo_id=repo_name, repo_type="dataset")
        print(f"Upload complete: {repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process real-world videos for tracking.")
    parser.add_argument("--input_dir", type=str, default=r"/scratch-shared/tnijdam/resized_videos")
    parser.add_argument("--output_dir", type=str, default=r"/scratch-shared/tnijdam/sam2_tracking")
    
    parser.add_argument("--real_world", action="store_true", help="Process real-world videos.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached results if available.")
    parser.add_argument("--upload_to_huggingface", action="store_true", help="Upload results to Hugging Face.")
    
    args = parser.parse_args()
    print(f"Processing videos in {args.input_dir}")
    process_all_videos(args)