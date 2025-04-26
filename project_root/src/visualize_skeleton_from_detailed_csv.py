import os
import csv
import cv2
import numpy as np
from collections import defaultdict

# Определяем связи между точками для рисования "скелета"
SKELETON_CONNECTIONS = [
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE'),
]

TORSO_POINTS = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']

def denormalize_and_scale(keypoints_norm, bbox):
    """Перевод нормализованных координат в реальные координаты bbox."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    keypoints_scaled = {}
    for name, kp in keypoints_norm.items():
        keypoints_scaled[name] = (
            int(x1 + (kp['x_norm'] + 0.5) * width),
            int(y1 + (kp['y_norm'] + 0.5) * height)
        )
    return keypoints_scaled

def draw_skeleton(image, keypoints, draw_torso=True):
    """Рисует скелет по точкам."""
    # Нарисовать линии между ключевыми точками
    for start, end in SKELETON_CONNECTIONS:
        if start in keypoints and end in keypoints:
            cv2.line(image, keypoints[start], keypoints[end], (0, 255, 0), 2)

    # Нарисовать сами точки
    for name, (x, y) in keypoints.items():
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    # Нарисовать заштрихованный торс
    if draw_torso and all(pt in keypoints for pt in TORSO_POINTS):
        pts = np.array([keypoints[pt] for pt in TORSO_POINTS], np.int32)
        cv2.fillPoly(image, [pts], (255, 200, 200))

def visualize(csv_path, output_dir, min_visibility=0.5, clean_canvas=False, live=False):
    """Генерирует визуализации скелетов"""
    if live:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Здесь можно вставить реальное чтение позы через модель
            cv2.imshow('Skeleton Live', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # Чтение CSV
    data = defaultdict(list)
    bboxes = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['image_path']
            bbox = eval(row['bbox']) if 'bbox' in row and row['bbox'] else [0, 0, 1, 1]
            bboxes[img_path] = bbox
            if float(row['visibility']) >= min_visibility:
                data[img_path].append({
                    'landmark': row['landmark'],
                    'x_norm': float(row['x_norm']),
                    'y_norm': float(row['y_norm']),
                })

    for img_path, keypoints_list in data.items():
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        if clean_canvas:
            image = np.ones_like(image) * 255

        keypoints_norm = {kp['landmark']: {'x_norm': kp['x_norm'], 'y_norm': kp['y_norm']} for kp in keypoints_list}
        bbox = bboxes.get(img_path, [0, 0, image.shape[1], image.shape[0]])
        keypoints_scaled = denormalize_and_scale(keypoints_norm, bbox)

        draw_skeleton(image, keypoints_scaled)

        # Сохраняем изображение
        output_img_path = os.path.join(output_dir, os.path.basename(img_path))
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_img_path, image)
