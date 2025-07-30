from ultralytics import YOLO
import cv2
import os
from glob import glob

# === CONFIGURATION ===
model_path = "runs/detect/train/weights/best.pt"
input_folder = "one_egg"       # Used in 'image' mode
video_path = "videosample.mp4"  # Used in 'video' mode
output_folder = "outputs"
mode = "image"  # "image", "webcam", or "video"

os.makedirs(output_folder, exist_ok=True)
model = YOLO(model_path)

def draw_boxes(image, results):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else [None]*len(boxes)
    egg_count = len(boxes)
    box_thickness = 8  # Thicker border
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        # Draw confidence score (closeness to 1)
        if score is not None:
            label = f"{score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    # Place total count at top-left and bottom-left for visibility
    height, width = image.shape[:2]
    text = f"Total Eggs: {egg_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 6
    # Top-left
    cv2.putText(image, text, (20, 60), font, font_scale, (0, 0, 255), thickness)
    # Bottom-left
    cv2.putText(image, text, (20, height - 30), font, font_scale, (0, 0, 255), thickness)
    return image, egg_count

# === IMAGE FOLDER MODE ===
if mode == "image":
    image_paths = glob(os.path.join(input_folder, "*.[jJpP][pPnN][gG]"))
    image_paths += glob(os.path.join(input_folder, "*.[jJ][pP][eE][gG]"))
    for img_path in image_paths:
        image = cv2.imread(img_path)
        results = model(img_path)
        print(f"🔍 Processing {img_path}...")
        image, egg_count = draw_boxes(image, results)
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, f"counted_{filename}")
        cv2.imwrite(save_path, image)
        print(f"✅ Processed {filename} - Total Eggs: {egg_count}")

# === VIDEO MODE ===
elif mode == "video":
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, "output_video.mp4")
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame, egg_count = draw_boxes(frame, results)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        out.write(frame)
        cv2.imshow("Video Egg Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key to stop
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("🎬 Video processing done!")

# === WEBCAM MODE ===
elif mode == "webcam":
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame, egg_count = draw_boxes(frame, results)

        cv2.imshow("Webcam Egg Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📷 Webcam session ended.")

else:
    print("❌ Invalid mode. Use 'image', 'video', or 'webcam'.")