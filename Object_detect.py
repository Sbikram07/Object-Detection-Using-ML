from ultralytics import YOLO
import cv2
import os

# Load trained YOLO model
model = YOLO("CustomModel/best.pt")

# Input image path (you can change this to any image)
image_path = "D:/Object Detection Using ML/gth.jpg"

# Output folder
output_folder = "D:/Object Detection Using ML/outputImages"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Run detection
results = model(image_path, conf=0.25)

# Get annotated image
annotated_image = results[0].plot()

# Extract file name and extension
file_name, file_ext = os.path.splitext(os.path.basename(image_path))

# Save as "<input>_detected"
output_path = os.path.join(output_folder, f"{file_name}_detected{file_ext}")
cv2.imwrite(output_path, annotated_image)

print(f" Detection completed and saved as: {output_path}")
