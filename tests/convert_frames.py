import os
import cv2
import re

# Get the current directory
frame_dir = "/home/tnijdam/VGMs/tools/sam2/tests/real_world/holonomic_pendulum/video_5_fps30"

# List all PNG files
frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
print(f"Found {len(frame_files)} PNG files")
# Extract frame numbers dynamically
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

# Sort by extracted number
frame_files.sort(key=extract_number)

# Convert and rename frames
for frame_file in frame_files:
    frame_number = extract_number(frame_file)
    
    if frame_number == float('inf'):
        print(f"Skipping {frame_file} (no valid number)")
        continue

    output_filename = f"{frame_number:05d}.jpg"
    img = cv2.imread(os.path.join(frame_dir, frame_file))

    if img is None:
        print(f"Error: Could not read {frame_file}")
        continue

    cv2.imwrite(os.path.join(frame_dir, output_filename), img)
    print(f"Converted: {frame_file} -> {output_filename}")
