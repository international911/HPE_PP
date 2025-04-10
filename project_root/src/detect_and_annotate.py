import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

def load_yolo_model(model_path='yolo11n.pt'):
    model = YOLO(model_path)
    return model

def setup_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return pose

def calculate_center(keypoints):
    """Вычисляет центр тела как среднюю точку между левым и правым бедром."""
    left_hip = keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    center_x = (left_hip.x + right_hip.x) / 2
    center_y = (left_hip.y + right_hip.y) / 2
    return center_x, center_y

def calculate_unit_length(keypoints):
    """Вычисляет единичный отрезок как среднее расстояние между плечами и бедрами."""
    left_shoulder = keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

    distance1 = np.linalg.norm([left_shoulder.x - right_hip.x, left_shoulder.y - right_hip.y])
    distance2 = np.linalg.norm([right_shoulder.x - left_hip.x, right_shoulder.y - left_hip.y])

    unit_length = (distance1 + distance2) / 2
    return unit_length

def normalize_coordinates(keypoints, center_x, center_y, unit_length):
    """Нормализует координаты относительно нового центра и единичного отрезка."""
    normalized_keypoints = []
    for landmark in keypoints:
        x = landmark.x
        y = landmark.y
        x_norm = (x - center_x) / unit_length
        y_norm = (y - center_y) / unit_length
        normalized_keypoints.append({"x_norm": x_norm, "y_norm": y_norm})
    return normalized_keypoints

def process_images_in_directory(directory, yolo_model, pose):
    annotations = []
    class_label = os.path.basename(directory)

    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            results = yolo_model(image)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_image = image[y1:y2, x1:x2]
                    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(cropped_image_rgb)

                    if pose_results.pose_landmarks:
                        keypoints = pose_results.pose_landmarks.landmark

                        # Определение центра тела
                        center_x, center_y = calculate_center(keypoints)

                        # Определение единичного отрезка
                        unit_length = calculate_unit_length(keypoints)

                        # Нормализация координат
                        normalized_keypoints = normalize_coordinates(keypoints, center_x, center_y, unit_length)

                        # Добавление аннотаций с меткой класса
                        for keypoint in normalized_keypoints:
                            annotations.append({
                                "x_norm": keypoint["x_norm"],
                                "y_norm": keypoint["y_norm"],
                                "class": class_label
                            })

    return annotations
