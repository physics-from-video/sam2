import cv2
import os
import glob

# Define the video file and output directory
video_path = "/home/tnijdam/VGMs/tools/sam2/tests/real_world/double_pendulum_14_11_2024/video_0_fps30"  # Replace with your actual video file path
output_dir = "./tests/1"   # Directory to save extracted frames

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
jpeg_images = []

# Read frames and save them as JPEGs
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no more frames

    # Generate filename with 5-digit numbering (00000.jpg, 00001.jpg, etc.)
    filename = os.path.join(output_dir, f"{frame_count:05d}.jpg")
    
    # Save the frame as a JPEG image
    cv2.imwrite(filename, frame)
    
    # Store the filename
    jpeg_images.append(filename)
    
    frame_count += 1

# Release the video capture object
cap.release()

# Print the list of extracted JPEG images
print("Extracted JPEG images:")
for img in jpeg_images:
    print(img)
