import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import os

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
                                       

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
test_dir = "./tests"
output_dir = "./outputs"

# test_name = "1"
test_name = "real_world/holonomic_pendulum/video_5_fps30"

video_dir = os.path.join(test_dir, test_name)
output_dir = os.path.join(output_dir, test_name)

os.makedirs(output_dir, exist_ok=True)

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# cut away first 40 frames
# print(frame_names[40])
# frame_names = frame_names[40:]

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

tests_points = {"1": {"points": np.array([[550, 450], [394, 417], [365, 416]], dtype=np.float32),
                    "labels": np.array([1, 0, 0], np.int32),
                    "ann_frame_idx": 0,
                    "ann_obj_id": 1},
                "real_world/double_pendulum": 
                    # {"points": np.array([[586, 429], [604, 403], [562, 469], [795, 483], [480, 420], [551, 540], [549, 407], [563, 361], [595, 300], [611, 477]], dtype=np.float32),
                    # {"points": np.array([[586, 429], [795, 483], [480, 420], [551, 540], [549, 407], [563, 361], [595, 300], [611, 477]], dtype=np.float32),
                    {"points": np.array([[586, 429]], dtype=np.float32),
                    "labels": np.array([1], np.int32),
                    "ann_frame_idx": 0,
                    "ann_obj_id": 1},
                "real_world/holonomic_pendulum/video_5_fps30": 
                    # {"points": np.array([[586, 429], [604, 403], [562, 469], [795, 483], [480, 420], [551, 540], [549, 407], [563, 361], [595, 300], [611, 477]], dtype=np.float32),
                    # {"points": np.array([[586, 429], [795, 483], [480, 420], [551, 540], [549, 407], [563, 361], [595, 300], [611, 477]], dtype=np.float32),
                    {"points": np.array([[642, 608], [504, 593], [514, 572], [504, 575], [533, 568], [504, 564], [500, 572]], dtype=np.float32),
                    "labels": np.array([1, 1, 0, 0, 0, 0, 0], np.int32),
                    "ann_frame_idx": 0,
                    "ann_obj_id": 1}
                    }
                # "real_world": 
                #     {"points": np.array([[471, 527], [478, 594], [458, 585], [504, 593]], dtype=np.float32),
                #     "labels": np.array([1, 0, 0, 0], np.int32),
                #     "ann_frame_idx": 0,
                #     "ann_obj_id": 1}
                #     }

                # "real_world": 
                #     {"points": np.array([[478, 594]], dtype=np.float32),
                #     "labels": np.array([1], np.int32),
                #     "ann_frame_idx": 0,
                #     "ann_obj_id": 1}
                #     }
points = tests_points[test_name]["points"]
labels = tests_points[test_name]["labels"]
ann_frame_idx = tests_points[test_name]["ann_frame_idx"]
ann_obj_id = tests_points[test_name]["ann_obj_id"]

frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

second_test_points = {"real_world/double_pendulum": 
                    # {"points": np.array([[586, 429], [604, 403], [562, 469], [795, 483], [480, 420], [551, 540], [549, 407], [563, 361], [595, 300], [611, 477]], dtype=np.float32),
                    # {"points": np.array([[586, 429], [795, 483], [480, 420], [551, 540], [549, 407], [563, 361], [595, 300], [611, 477]], dtype=np.float32),
                    {"points": np.array([[562, 356]], dtype=np.float32),
                    "labels": np.array([1], np.int32),
                    "ann_frame_idx": 0,
                    "ann_obj_id": 2}
                    }

# only if there is a match, we add a second point, otherwise ignore
if test_name in second_test_points.keys():
    points = second_test_points[test_name]["points"]
    labels = second_test_points[test_name]["labels"]
    ann_frame_idx = second_test_points[test_name]["ann_frame_idx"]
    ann_obj_id = second_test_points[test_name]["ann_obj_id"]

    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

plt.savefig(f"{output_dir}/frame{ann_frame_idx}.jpg")
plt.close()

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
vis_frame_stride = 1
plt.close("all")

os.makedirs(f"{output_dir}/frames", exist_ok=True)
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    plt.savefig(f"{output_dir}/frames/frame{out_frame_idx}.jpg")
    # close 
    plt.close()
    
# create a video from the saved frames
import cv2
import os
output_video_path = f"{output_dir}/output.mp4"
frame_dir = f"{output_dir}/frames"

frame_names = [
    p for p in os.listdir(frame_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]

# Extract numeric part safely
def extract_frame_number(filename):
    try:
        return int(os.path.splitext(filename)[0][5:])  # Extract number from "frameXXXXX.jpg"
    except ValueError:
        return float('inf')  # Push non-numeric names to the end

# Sort frames based on extracted numbers
frame_names.sort(key=extract_frame_number)

# Read first image to get dimensions
first_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
height, width, _ = first_frame.shape

# Define the video writer
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

# Write frames to video
for frame_name in frame_names:
    if "frame" in frame_name:  # Ensure we only process numbered frames
        img = cv2.imread(os.path.join(frame_dir, frame_name))
        video.write(img)

# Release resources
video.release()
cv2.destroyAllWindows()










# The code above is equivalent to the following pseudo code:
# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     state = predictor.init_state(<your_video>)


# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     state = predictor.init_state(<your_video>)

#     # add new prompts and instantly get the output on the same frame
#     frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

#     # propagate the prompts to get masklets throughout the video
#     for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#         ...