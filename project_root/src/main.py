from src.detect_and_annotate import load_yolo_model, setup_mediapipe, detect_and_annotate, save_annotations

def main():
    yolo_model = load_yolo_model('models/yolov11.pt')
    pose = setup_mediapipe()
    image_path = 'data/images/person_1.jpg'
    annotations = detect_and_annotate(image_path, yolo_model, pose)
    output_path = 'data/keypoints.json'
    save_annotations(annotations, output_path)
    print(f"Annotation save {output_path}")

if __name__ == "__main__":
    main()
