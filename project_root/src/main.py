import os
import pandas as pd
from src.detect_and_annotate import load_yolo_model, setup_mediapipe, process_images_in_directory, save_annotations_to_csv
from src.cnn_model import train_cnn_model

def main(directory):
    if not os.path.exists(directory):
        print(f"Директория {directory} не существует.")
        return

    # Загрузка модели YOLO
    yolo_model = load_yolo_model('yolo11n.pt')

    # Настройка MediaPipe
    pose = setup_mediapipe()

    annotations = process_images_in_directory(directory, yolo_model, pose)

    output_path = 'data/keypoints.csv'
    save_annotations_to_csv(annotations, output_path)
    print(f"Кейпоинты сохранены в {output_path}")

    if os.path.exists(output_path):
        data = pd.read_csv(output_path)
        print("Первые несколько строк CSV файла:")
        print(data.head())
    else:
        print(f"Файл {output_path} не найден.")

    # Обучение CNN-модели
    train_cnn_model(output_path)

if __name__ == "__main__":
    main('data/images')
