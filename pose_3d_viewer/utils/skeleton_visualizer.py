import cv2
import numpy as np

class SkeletonVisualizer:
    def __init__(self, connections):
        self.connections = connections
        self.img_size = (1000, 1000)
        
        # Цвета для разных частей тела
        self.colors = {
            'face': (255, 0, 0),      # Красный
            'arms': (0, 255, 0),      # Зеленый
            'legs': (0, 0, 255),      # Синий
            'torso': (255, 255, 0)    # Желтый
        }

    def draw(self, joints):
        """Создает 2D изображение скелета"""
        try:
            # Создаем белое изображение
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            img.fill(255)
            
            # Преобразуем координаты в пиксели
            pixel_joints = {}
            for name, (x, y, _) in joints.items():
                px = int((x + 2.5) * self.img_size[0] / 5)  # Масштабирование для вашего диапазона координат
                py = int((y + 2.5) * self.img_size[1] / 5)
                pixel_joints[name] = (px, py)
            
            # Рисуем соединения
            for start, end in self.connections:
                if start in pixel_joints and end in pixel_joints:
                    color = self._get_connection_color(start)
                    cv2.line(img, pixel_joints[start], pixel_joints[end], color, 3)
            
            # Рисуем суставы
            for point in pixel_joints.values():
                cv2.circle(img, point, 5, (0, 0, 0), -1)  # Черные точки
            
            return img
        
        except Exception as e:
            print(f"Ошибка при рисовании скелета: {str(e)}")
            return None

    def _get_connection_color(self, joint):
        """Определяет цвет соединения на основе имени сустава"""
        if "EYE" in joint or "NOSE" in joint or "MOUTH" in joint or "EAR" in joint:
            return self.colors['face']
        elif "SHOULDER" in joint or "ELBOW" in joint or "WRIST" in joint or "PINKY" in joint or "INDEX" in joint or "THUMB" in joint:
            return self.colors['arms']
        elif "HIP" in joint or "KNEE" in joint or "ANKLE" in joint or "HEEL" in joint or "FOOT" in joint:
            return self.colors['legs']
        else:
            return self.colors['torso']