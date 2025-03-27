import os
from src.detect_and_annotate import load_yolo_model, setup_mediapipe, detect_and_annotate, save_annotations

def main():
    yolo_model = load_yolo_model('yolo11n.pt')

    pose = setup_mediapipe()

    images_folder = 'data/images'

    all_annotations = {}
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, filename)
            print(f"Обработка изображения: {image_path}")

            annotations = detect_and_annotate(image_path, yolo_model, pose)

            all_annotations[filename] = annotations

    output_path = 'data/keypoints.json'
    save_annotations(all_annotations, output_path)
    print(f"Keypoints save {output_path}")

if __name__ == "__main__":
    main()
