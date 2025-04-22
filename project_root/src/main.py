import os
import argparse
from pathlib import Path
from detect_and_annotate import PoseProcessor

def get_project_root() -> Path:
    """Возвращает абсолютный путь к корню проекта"""
    return Path(__file__).parent.parent

def process_dataset(dataset_root: str = None):
    # Получаем правильные пути
    project_root = get_project_root()
    dataset_root = dataset_root or str(project_root / 'data' / 'dataset')
    output_dir = project_root / 'data' / 'annotations'
    
    processor = PoseProcessor(yolo_model_path=str(project_root / 'yolo11n.pt'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_paths = []
    for class_name in ['standing', 'sitting', 'lying']:
        class_dir = os.path.join(dataset_root, class_name)
        if os.path.exists(class_dir):
            output_csv = output_dir / f'{class_name}_keypoints.csv'
            processor.process_class_directory(class_dir, str(output_csv))
            csv_paths.append(str(output_csv))
            print(f"Processed {class_name} -> {output_csv}")
    
    return csv_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to dataset directory')
    parser.add_argument('--train', action='store_true', help='Train model after processing')
    args = parser.parse_args()

    # Обработка данных
    print("Processing dataset...")
    csv_files = process_dataset(args.dataset)
    
    # Обучение модели (если нужно)
    if args.train and csv_files:
        print("\nTraining model...")
        from cnn_model import train_model
        project_root = get_project_root()
        model_path = project_root / 'models' / 'pose_classifier.pth'
        train_model(csv_files, str(model_path))
        print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    main()