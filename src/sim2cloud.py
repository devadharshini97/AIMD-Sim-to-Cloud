import os
import cv2
import json
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

print("\nStep 1: Installing YOLOv8...")
os.system('pip install ultralytics -q')
print("\nStep 2: Setting up paths...")

image_dir = '/home/sagemaker-user/shared/isaac_render_product/rgb'
output_dir = './yolo_detections'
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(f'{image_dir}/*.png'))
print(f"Found {len(image_paths)} images")

if len(image_paths) == 0:
    print(f"ERROR: No images found in {image_dir}")
    exit(1)

print(f"Sample images:")
for img in image_paths[:3]:
    print(f"  - {os.path.basename(img)}")

print("\nStep 3: Loading pretrained YOLOv8 model...")

model_name = f'yolov8{model_size}.pt'
print(f"Loading YOLOv8{model_size} (pretrained on COCO)...")
model = YOLO(model_name)  # Auto-downloads if not present
print(f" - Model loaded: {model_name}")
print(f"  - Architecture: YOLOv8{model_size}")
print(f"  - Pretrained on: COCO (80 object classes)")
print(f"  - Classes: person, car, dog, cat, bicycle, etc.")

print("\nStep 4: Running inference on all images...")

results_list = []
detections_dict = {}

for idx, image_path in enumerate(image_paths):
    filename = os.path.basename(image_path)
    results = model(image_path, verbose=False)
    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            detection = {
                'class_id': int(box.cls.item()),
                'class_name': result.names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'bbox': {
                    'x_min': float(box.xyxy[0, 0].item()),
                    'y_min': float(box.xyxy[0, 1].item()),
                    'x_max': float(box.xyxy[0, 2].item()),
                    'y_max': float(box.xyxy[0, 3].item()),
                    'width': float(box.xywh[0, 2].item()),
                    'height': float(box.xywh[0, 3].item()),
                }
            }
            detections.append(detection)
    detections_dict[filename] = {
        'num_detections': len(detections),
        'detections': detections,
        'image_path': image_path
    }
    results_list.append(result)
    
    if (idx + 1) % max(1, len(image_paths) // 10) == 0:
        print(f"  Processed {idx + 1}/{len(image_paths)} images...")

print(f"✓ Inference complete on all {len(image_paths)} images")

print("\nStep 5: Saving detection results...")

detections_json_path = os.path.join(output_dir, 'detections.json')
with open(detections_json_path, 'w') as f:
    json.dump(detections_dict, f, indent=2)
print(f"✓ Saved detections to: {detections_json_path}")

print("\nStep 6: Detection Summary Statistics")

total_detections = sum(item['num_detections'] for item in detections_dict.values())
images_with_detections = sum(1 for item in detections_dict.values() 
                            if item['num_detections'] > 0)

print(f"Total images processed: {len(image_paths)}")
print(f"Total objects detected: {total_detections}")
print(f"Images with detections: {images_with_detections}")
print(f"Average detections per image: {total_detections / len(image_paths):.2f}")

class_counts = {}
for item in detections_dict.values():
    for detection in item['detections']:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

print(f"\nDetected object classes:")
for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {class_name}: {count} detections")

print("\nStep 7: Creating visualizations...")

num_samples = min(3, len(image_paths))
fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

if num_samples == 1:
    axes = [axes]

for idx in range(num_samples):
    image_path = image_paths[idx]
    filename = os.path.basename(image_path)
    
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Plot image
    axes[idx].imshow(img)
    axes[idx].set_title(f'{filename}\n({detections_dict[filename]["num_detections"]} objects)')
    
    # Draw bounding boxes
    for detection in detections_dict[filename]['detections']:
        bbox = detection['bbox']
        x_min, y_min = bbox['x_min'], bbox['y_min']
        width = bbox['x_max'] - bbox['x_min']
        height = bbox['y_max'] - bbox['y_min']
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[idx].add_patch(rect)
        
        # Add label
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        axes[idx].text(x_min, y_min - 10, label, 
                      color='red', fontsize=8, 
                      bbox=dict(facecolor='yellow', alpha=0.5))
    
    axes[idx].axis('off')

plt.tight_layout()
viz_path = os.path.join(output_dir, 'sample_detections.png')
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {viz_path}")
plt.close()

print("\nStep 8: Saving annotated images...")

annotated_dir = os.path.join(output_dir, 'annotated_images')
os.makedirs(annotated_dir, exist_ok=True)

for idx, (image_path, result) in enumerate(zip(image_paths, results_list)):
    filename = os.path.basename(image_path)
    
    # Get annotated image from YOLO
    annotated_img = result.plot()
    
    # Save annotated image
    output_path = os.path.join(annotated_dir, filename)
    cv2.imwrite(output_path, annotated_img)
    
    if (idx + 1) % max(1, len(image_paths) // 10) == 0:
        print(f"  Saved {idx + 1}/{len(image_paths)} annotated images...")

print(f"All annotated images saved to: {annotated_dir}")

print("\nStep 9: Creating summary report...")

report = f"""
YOLO Object Detection Results
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Images:
  Directory: {image_dir}
  Total images: {len(image_paths)}

Model:
  Architecture: YOLOv8{model_size}
  Pretrained: COCO dataset (80 classes)
  Classes: person, car, dog, cat, bicycle, truck, etc.

Detection Summary:
  Total objects detected: {total_detections}
  Images with detections: {images_with_detections}
  Average objects per image: {total_detections / len(image_paths):.2f}

Detected Classes:
"""

for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    report += f"  - {class_name}: {count} detections\n"

report += f"""

Output Files:
  - {detections_json_path}
  - {annotated_dir}/
  - {viz_path}
  - {os.path.join(output_dir, 'report.txt')}
"""

report_path = os.path.join(output_dir, 'report.txt')
with open(report_path, 'w') as f:
    f.write(report)

print(report)
print(f"Report saved to: {report_path}")

print("\nStep 10: Upload results to S3 (Optional)")
print("-" * 70)

import sagemaker
import boto3

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
s3_prefix = f'isaac-sim/yolo-detections/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}'

response = input(f"\nUpload results to S3? (y/n) [s3://{bucket}/{s3_prefix}]: ").strip().lower()

if response == 'y':
    s3_client = boto3.client('s3')
    
    # Upload detections JSON
    s3_client.upload_file(detections_json_path, bucket, f'{s3_prefix}/detections.json')
    print(f"Uploaded detections.json")
    
    # Upload report
    s3_client.upload_file(report_path, bucket, f'{s3_prefix}/report.txt')
    print(f"Uploaded report.txt")
    
    # Upload sample visualization
    s3_client.upload_file(viz_path, bucket, f'{s3_prefix}/sample_detections.png')
    print(f"Uploaded sample_detections.png")
    
    # Upload all annotated images
    for img_file in os.listdir(annotated_dir):
        img_path = os.path.join(annotated_dir, img_file)
        if os.path.isfile(img_path):
            s3_client.upload_file(img_path, bucket, f'{s3_prefix}/annotated_images/{img_file}')
    
    print(f"\n✓ All results uploaded to: s3://{bucket}/{s3_prefix}/")
else:
    print("Skipped S3 upload")

print("INFERENCE COMPLETE!")
print(f"\nLocal results saved to: {output_dir}/")
print(f"  - detections.json       (all bounding boxes)")
print(f"  - annotated_images/     (visualized detections)")
print(f"  - sample_detections.png (preview)")
print(f"  - report.txt            (summary statistics)")
