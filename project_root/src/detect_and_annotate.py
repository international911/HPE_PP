import cv2
import json
from ultralytics import YOLO
import mediapipe as mp

def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

def setup_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return pose

def detect_and_annotate(image_path, yolo_model, pose):
    image = cv2.imread(image_path)
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
                    keypoints.append({"x": x, "y": y})
                annotations.append(keypoints)

    return annotations

def save_annotations(annotations, output_path):
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)
