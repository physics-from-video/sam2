import json
import re
from skimage.measure import label, regionprops
import numpy as np
import os
from utils.plotting import overlay_mask_on_image
from PIL import Image
import pickle

def transform_keys(data):
    """Recursively transform keys to retain only 'video_X' format."""
    new_data = {}
    for key, value in data.items():
        match = re.match(r"(video_\d+)", key)
        new_key = match.group(1) if match else key
        if new_key in new_data and isinstance(new_data[new_key], dict) and isinstance(value, dict):
            new_data[new_key].update(value)
        else:
            new_data[new_key] = transform_keys(value) if isinstance(value, dict) else value
    return new_data

def load_labels(labels_path):
    """Load and transform labels from a JSON file."""
    with open(labels_path, "r") as f:
        data = json.load(f)
    return transform_keys(data)


def compute_mask_properties(mask, experiment):
    """
    Compute the centroid (centre) of a mask, and—for holonomic pendulum—its
    maximum distance (using regionprops' major_axis_length as a proxy).
    """
    mask = np.squeeze(mask)
    mask_bool = mask > 0
    indices = np.argwhere(mask_bool)
    centre = tuple(np.mean(indices, axis=0)) if indices.size > 0 else (None, None)
    max_distance = None
    if experiment == "holonomic_pendulum" and indices.size > 0:
        labeled = label(mask_bool)
        props = regionprops(labeled)
        max_distance = props[0].major_axis_length if props else 0.0
    return centre, max_distance

def process_object_points(label_info):
    """Convert label info into a dictionary of object points and labels."""
    object_points = {}
    for obj_key, obj_data in label_info.items():
        obj_id = int(obj_key.split("_")[-1])
        pos = np.array(obj_data.get("positive", []), dtype=np.float32)
        neg = np.array(obj_data.get("negative", []), dtype=np.float32)
        pts, labs = [], []
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
    return object_points

def process_segmentation_frames(frames_dir, video_segments, tracking_plots_dir, experiment):
    """
    Process and save segmentation overlays for every 10th frame.
    Computes centres and, if applicable, max distances over time.
    """
    # Sort frame names numerically.
    frame_names = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=lambda p: int(os.path.splitext(p)[0])
    )
    centers_over_time = {}       # {obj_id: [(frame_idx, (y, x)), ...]}
    masks_over_time = {}         # {frame_idx: {obj_id: mask}}
    max_distance_over_time = {}  # {obj_id: [(frame_idx, max_distance), ...]} for holonomic pendulum

    # Ensure output directory for frames exists.
    frames_output = os.path.join(tracking_plots_dir, "frames")
    os.makedirs(frames_output, exist_ok=True)

    save_every = 10
    
    for out_frame_idx in range(0, len(frame_names)):
        frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        frame_img = np.array(Image.open(frame_path))
        if out_frame_idx in video_segments:
            masks_over_time[out_frame_idx] = {}
            for obj_id, mask in video_segments[out_frame_idx].items():
                frame_img = overlay_mask_on_image(frame_img, mask, obj_id=obj_id)
                centre, max_distance = compute_mask_properties(mask, experiment)
                masks_over_time[out_frame_idx][obj_id] = mask
                if centre[0] is not None:
                    centers_over_time.setdefault(obj_id, []).append((out_frame_idx, centre))
                if experiment == "holonomic_pendulum":
                    max_distance_over_time.setdefault(obj_id, []).append((out_frame_idx, max_distance))
        if out_frame_idx % save_every == 0:
            save_path = os.path.join(frames_output, f"frame_{out_frame_idx:05d}_segmentation.jpg")
            Image.fromarray(frame_img).save(save_path)
    return centers_over_time, masks_over_time, max_distance_over_time

def save_tracking_data(output_folder, centers_over_time, masks_over_time, max_distance_over_time, experiment):
    """Save computed centres, masks, and (if applicable) max distances as pickle files."""
    with open(os.path.join(output_folder, "centres.pkl"), "wb") as f:
        pickle.dump(centers_over_time, f)
    with open(os.path.join(output_folder, "masks.pkl"), "wb") as f:
        pickle.dump(masks_over_time, f)
    if experiment == "holonomic_pendulum":
        with open(os.path.join(output_folder, "max_distance.pkl"), "wb") as f:
            pickle.dump(max_distance_over_time, f)
