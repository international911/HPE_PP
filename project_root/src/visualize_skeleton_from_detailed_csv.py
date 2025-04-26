import os
import csv
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

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

TORSO_POINTS = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'LEFT_HIP']

def denormalize_keypoints(keypoints_norm, center_x, center_y, unit_length, image_shape):
    height, width = image_shape[:2]
    keypoints_scaled = {}
    for name, kp in keypoints_norm.items():
        x = int((center_x + kp['x_norm'] * unit_length) * width)
        y = int((center_y + kp['y_norm'] * unit_length) * height)
        keypoints_scaled[name] = (x, y)
    return keypoints_scaled

def draw_skeleton(image, keypoints, draw_torso=True):
    for start, end in SKELETON_CONNECTIONS:
        if start in keypoints and end in keypoints:
            cv2.line(image, keypoints[start], keypoints[end], (0, 255, 0), 2)

    for name, (x, y) in keypoints.items():
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    if draw_torso and all(pt in keypoints for pt in TORSO_POINTS):
        pts = np.array([keypoints[pt] for pt in TORSO_POINTS], np.int32)
        cv2.fillPoly(image, [pts], (255, 200, 200))

def visualize(
    csv_path=None, 
    output_dir=None, 
    min_visibility=0.5, 
    clean_canvas=False, 
    live=False, 
    sample_size=None
):
    if live:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Здесь должна быть логика обработки live-видео
            cv2.imshow('Live Pose Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    data = defaultdict(list)
    meta = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['image_path']
            if float(row['visibility']) >= min_visibility:
                data[img_path].append({
                    'landmark': row['landmark'],
                    'x_norm': float(row['x_norm']),
                    'y_norm': float(row['y_norm'])
                })
                meta[img_path] = {
                    'center_x': float(row['center_x']),
                    'center_y': float(row['center_y']),
                    'unit_length': float(row['unit_length'])
                }

    img_paths = list(data.keys())
    if sample_size is not None and sample_size > 0:
        img_paths = img_paths[:sample_size]

    for img_path in tqdm(img_paths, desc="Visualizing skeletons"):
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image {img_path}, skipping.")
            continue

        if clean_canvas:
            image = np.ones_like(image) * 255

        keypoints_norm = {kp['landmark']: {'x_norm': kp['x_norm'], 'y_norm': kp['y_norm']} 
                         for kp in data[img_path]}
        meta_info = meta[img_path]
        keypoints_scaled = denormalize_keypoints(
            keypoints_norm, 
            meta_info['center_x'], 
            meta_info['center_y'], 
            meta_info['unit_length'], 
            image.shape
        )

        draw_skeleton(image, keypoints_scaled)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_img_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_img_path, image)
        else:
            cv2.imshow('Skeleton Visualization', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--clean_canvas', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--sample_size', type=int, default=None)
    args = parser.parse_args()
    
    visualize(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        clean_canvas=args.clean_canvas,
        live=args.live,
        sample_size=args.sample_size
    )