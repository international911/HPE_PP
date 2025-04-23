import sys
import json
import cv2
import numpy as np
from pathlib import Path
from utils.simple_3d_viewer import Model3DViewer
from utils.skeleton_visualizer import SkeletonVisualizer

class PoseViewer:
    def __init__(self):
        # Загрузка конфигурации соединений
        self.config_path = Path(__file__).parent / "config" / "skeleton_connections.json"
        with open(self.config_path) as f:
            self.connections = json.load(f)["connections"]
        
        # Инициализация визуализаторов
        self.model_3d = Model3DViewer(self.connections)
        self.visualizer_2d = SkeletonVisualizer(self.connections)

    def load_keypoints(self, csv_path):
        """Загрузка ключевых точек из CSV файла"""
        joints = {}
        try:
            with open(csv_path) as f:
                for line in f:
                    # Пропускаем строки без достаточного количества значений
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        try:
                            name = parts[1]
                            x, y, z = map(float, parts[2:5])
                            joints[name] = [x, y, z]
                        except (ValueError, IndexError):
                            continue
            return joints
        except FileNotFoundError:
            print(f"Ошибка: Файл не найден: {csv_path}")
            return None
        except Exception as e:
            print(f"Ошибка при чтении файла: {str(e)}")
            return None

    def visualize(self, csv_path):
        """Основная функция визуализации"""
        # Проверяем существование файла
        if not Path(csv_path).exists():
            print(f"Файл с аннотациями не найден по пути: {csv_path}")
            print("Убедитесь, что указали правильный путь к файлу .csv")
            return
        
        joints = self.load_keypoints(csv_path)
        if not joints:
            print("Не удалось загрузить ключевые точки")
            return
        
        print(f"Успешно загружено {len(joints)} ключевых точек")
        
        # 2D визуализация
        img = self.visualizer_2d.draw(joints)
        if img is not None:
            cv2.imshow("2D Human Model", img)
            cv2.waitKey(3000)  # Показываем 3 секунды
            cv2.destroyAllWindows()
        
        # 3D визуализация
        self.model_3d.plot(joints)

if __name__ == "__main__":
    viewer = PoseViewer()
    
    # Укажите абсолютный путь к вашему файлу с аннотациями
    csv_path = "/Users/timur/Desktop/Вуз/Project_sem3/HPE_PP/project_root/data/annotations/standing_keypoints.csv"
    
    # Альтернативный вариант - относительный путь
    # csv_path = Path(__file__).parent.parent / "project_root" / "data" / "annotations" / "standing_keypoints.csv"
    
    viewer.visualize(csv_path)