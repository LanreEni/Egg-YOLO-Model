# 🥚 Egg Detection using YOLOv8

A computer vision project for detecting eggs using the **YOLOv8** object detection model.  
This project uses a **custom-trained model** to identify and locate eggs in images, videos, folders, or real-time webcam streams.

---

## 📁 Project Structure


📦 egg-detection-yolov8

├── yolov8n.pt            # Trained YOLOv8 model weights

├── yolo8_detect.py       # Main script for running detections

├── README.md             # Project documentation


└── requirements.txt      # Python dependencies


---

## 🔧 Setup Instructions

1. Clone the repository
   
bash
git clone https://github.com/your-username/egg-detection-yolov8.git
cd egg-detection-yolov8

# Install dependencies

pip install -r requirements.txt
Add this to your requirements.txt if it’s not already there:

nginx
Copy
Edit
ultralytics
opencv-python

# 🧪 How to Use
---
Run the detection script:

python yolo8_detect.py

Then follow the on-screen instructions to:

Choose input type (image, folder, video, webcam)

Provide the path to your media file or stream

The predictions will be saved to: runs/detect/predict/


# 🧠 Model Details
---
Model: YOLOv8n (nano) trained on custom egg dataset

Training Tool: Roboflow

Classes Detected: ['egg']

Input Types Supported: image, folder, video, webcam

# 📊 Results Sample
Media	Detection
Image	✅
Video	✅
Webcam	✅

📷 Sample Output
(Insert screenshots or short GIFs here showing detections on eggs)

🙌 Acknowledgements
Ultralytics YOLOv8

Roboflow

OpenCV

📌 TODOs
 Real-time detection counter

 Filter by confidence threshold

 Deploy on a web app
