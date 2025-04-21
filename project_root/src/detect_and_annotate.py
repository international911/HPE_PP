import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from tqdm import tqdm

class PoseProcessor:
    def __init__(self, yolo_model_path='yolo11n.pt'):
        self.yolo = YOLO(yolo_model_path)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.landmark_names = [
            name for name in self.mp_pose.PoseLandmark._member_names_
        ]

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None

        results = self.yolo(image)
        keypoints_data = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped = image[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(cropped_rgb)

                if pose_results.pose_landmarks:
                    keypoints = self._normalize_keypoints(pose_results.pose_landmarks.landmark)
                    keypoints_data.append({
                        'image_path': image_path,
                        'bbox': [x1, y1, x2, y2],
                        'keypoints': keypoints
                    })

        return keypoints_data

    def _normalize_keypoints(self, landmarks):
        # Получаем координаты бедер для центра
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        center_x = (left_hip.x + right_hip.x) / 2
        center_y = (left_hip.y + right_hip.y) / 2

        # Вычисляем масштабный коэффициент (расстояние между плечами)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        unit_length = np.sqrt(
            (right_shoulder.x - left_shoulder.x)**2 + 
            (right_shoulder.y - left_shoulder.y)**2
        )

        # Нормализуем все ключевые точки
        normalized = {}
        for i, landmark in enumerate(landmarks):
            normalized[self.landmark_names[i]] = {
                'x_norm': (landmark.x - center_x) / unit_length,
                'y_norm': (landmark.y - center_y) / unit_length,
                'visibility': landmark.visibility
            }
        return normalized

    def process_class_directory(self, class_dir, output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['image_path', 'landmark', 'x_norm', 'y_norm', 'visibility']
            writer.writerow(header)

            for img_file in tqdm(os.listdir(class_dir), desc=f"Processing {os.path.basename(class_dir)}"):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    results = self.process_image(img_path)
                    
                    if results:
                        for result in results:
                            for landmark_name, coords in result['keypoints'].items():
                                writer.writerow([
                                    img_path,
                                    landmark_name,
                                    coords['x_norm'],
                                    coords['y_norm'],
                                    coords['visibility']
                                ])