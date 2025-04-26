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

        self.required_landmarks = [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_HIP', 'RIGHT_HIP'
        ]

        self.optional_landmarks = [
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_EYE', 'RIGHT_EYE'
        ]

        self.landmark_names = self.required_landmarks + self.optional_landmarks

    def _calculate_body_center(self, landmarks):
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]
        x_coords = [landmarks[kp.value].x for kp in key_points]
        y_coords = [landmarks[kp.value].y for kp in key_points]
        return np.mean(x_coords), np.mean(y_coords)

    def _calculate_unit_length(self, landmarks):
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rb = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lb = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]

        diag1 = np.hypot(ls.x - rb.x, ls.y - rb.y)
        diag2 = np.hypot(rs.x - lb.x, rs.y - lb.y)
        return max((diag1 + diag2) / 2, 0.01)

    def _normalize_keypoints(self, landmarks):
        center_x, center_y = self._calculate_body_center(landmarks)
        unit_length = self._calculate_unit_length(landmarks)

        normalized = {}
        for name in self.landmark_names:
            idx = getattr(self.mp_pose.PoseLandmark, name).value
            landmark = landmarks[idx]
            if landmark.visibility < 0.5:
                continue

            normalized[name] = {
                'x_norm': (landmark.x - center_x) / unit_length,
                'y_norm': (landmark.y - center_y) / unit_length,
                'visibility': landmark.visibility
            }

        # Проверка обязательных точек
        for name in self.required_landmarks:
            if name not in normalized:
                return None

        return normalized

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
                if cropped.size == 0:
                    continue
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(cropped_rgb)

                if pose_results.pose_landmarks:
                    keypoints = self._normalize_keypoints(pose_results.pose_landmarks.landmark)
                    if keypoints:
                        keypoints_data.append({
                            'image_path': image_path,
                            'bbox': [x1, y1, x2, y2],
                            'keypoints': keypoints
                        })
        return keypoints_data

    def process_class_directory(self, class_dir, output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'bbox', 'landmark', 'x_norm', 'y_norm', 'visibility', 'class'])

            class_name = os.path.basename(class_dir)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
                img_path = os.path.join(class_dir, img_file)
                results = self.process_image(img_path)

                if results:
                    for result in results:
                        for landmark_name, coords in result['keypoints'].items():
                            writer.writerow([
                                result['image_path'],
                                result['bbox'],
                                landmark_name,
                                coords['x_norm'],
                                coords['y_norm'],
                                coords['visibility'],
                                class_name
                            ])
