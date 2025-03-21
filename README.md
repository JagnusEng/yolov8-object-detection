# 🧠 YOLOv8 Object Detection on COCO Dataset + Custom Image Test

This project demonstrates real-time object detection using **YOLOv8** (You Only Look Once, Version 8) in **Google Colab**. I used the **COCO 2017 validation dataset** to train and test a YOLOv8 model, and also tested the model on a **custom image** to simulate a real-world example.

---

## 📸 Project Features

- ✅ Trained a pre-trained YOLOv8 model on the COCO128 dataset (subset of COCO)
- ✅ Detected and classified objects from the COCO validation set
- ✅ Uploaded and tested a custom image (with two dogs) using the trained model
- ✅ Displayed results with bounding boxes and confidence scores
- ✅ Exported the trained model in ONNX format
- ✅ Resized display output for cleaner viewing in Colab

---

## 🚀 Tech Stack

| Tool/Library   | Purpose                        |
|----------------|--------------------------------|
| `YOLOv8`       | Object detection (Ultralytics) |
| `OpenCV`       | Image processing and loading   |
| `Matplotlib`   | Image visualization            |
| `Google Colab` | Cloud-based Python notebook    |
| `COCO`         | Dataset for training/testing   |

---

## 📂 Dataset Used

- **COCO 2017 Validation Set** (`val2017.zip`)  
  Downloaded and extracted directly into the Colab environment.  
  Source: [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)

- **COCO128.yaml**  
  A smaller version of the COCO dataset provided by Ultralytics, ideal for fast training.

---

## 🧪 Model Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Nano model (fast and lightweight)
model.train(data="coco128.yaml", epochs=3, imgsz=640)

Trained on 640x640 images for 3 epochs
Used yolov8n.pt, a lightweight model good for fast prototyping

## Real-World Image Test
- **After training, I uploaded and tested my own image (with 2 dogs). Here's what I did:**
from google.colab import files
uploaded = files.upload()  # Upload any image

image_path = next(iter(uploaded))
results = model(image_path)
results[0].show()


## 🖼️ Resized Display
import matplotlib.pyplot as plt
import cv2

image_with_boxes = results[0].plot()

plt.figure(figsize=(6, 6))  # Smaller display
plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Smaller Display of Detection")
plt.show()


## 🎯 Results & Insights
YOLOv8 was able to detect objects in both COCO images and a real-world uploaded image.
Confidence scores (e.g., 0.62, 0.45) were lower for the custom image due to:
The small YOLOv8n model being less accurate
Variations in lighting, object position, and background
Results were still accurate enough to demonstrate object detection capabilities.


