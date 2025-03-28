import cv2
import json
from ultralytics import YOLO
import mediapipe as mp

def load_yolo_model(model_path='yolo11n.pt'):
    try:
        model = YOLO(model_path)
        print("Модель загружена.")
        return model
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def setup_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return pose

def normalize_coordinates(x, y, width, height):
    return x / width, y / height

def detect_and_annotate(image_path, yolo_model, pose):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    results = yolo_model(image)

    annotations = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(cropped_image_rgb)

            if pose_results.pose_landmarks:
                keypoints = []
                for landmark in pose_results.pose_landmarks.landmark:
                    x = int(landmark.x * (x2 - x1)) + x1
                    y = int(landmark.y * (y2 - y1)) + y1
                    x_norm, y_norm = normalize_coordinates(x, y, width, height)
                    keypoints.append({"x": x, "y": y, "x_norm": x_norm, "y_norm": y_norm})
                annotations.append(keypoints)

    return annotations

def save_annotations(annotations, output_path):
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)
