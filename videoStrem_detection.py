from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Parameters
# -----------------------------
model_path = "customModel/best.pt"  # Path to trained YOLO model
input_video = "D:/Object Detection Using ML/inputVideos/240p1.mp4"  # Video file path
output_folder = "D:/Object Detection Using ML/outputVideos"
conf_threshold = 0.25  # Detection confidence threshold


# Load YOLO model
model = YOLO(model_path)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video capture
cap = cv2.VideoCapture(input_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Extract video name and extension
video_name, video_ext = os.path.splitext(os.path.basename(input_video))

# Output path with "_detected"
output_path = os.path.join(output_folder, f"{video_name}_detected{video_ext}")

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Starting video detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame, conf=conf_threshold)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Write frame to output video
    out.write(annotated_frame)

print(f" Detection complete. Output saved at: {output_path}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
