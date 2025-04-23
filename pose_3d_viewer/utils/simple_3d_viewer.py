import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Model3DViewer:
    def __init__(self, connections):
        self.connections = connections
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Настройки 3D сцены
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(-3, 3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Цвета для разных частей тела
        self.colors = {
            'face': 'red',
            'arms': 'green',
            'legs': 'blue',
            'torso': 'yellow'
        }

    def plot(self, joints):
        self.ax.clear()
        
        # Рисуем соединения
        for start, end in self.connections:
            if start in joints and end in joints:
                color = self._get_connection_color(start)  # Передаем только один сустав
                x = [joints[start][0], joints[end][0]]
                y = [joints[start][1], joints[end][1]]
                z = [joints[start][2], joints[end][2]]
                self.ax.plot(x, y, z, color=color, linewidth=3)
        
        # Рисуем суставы
        for name, (x, y, z) in joints.items():
            self.ax.scatter(x, y, z, c='black', s=50)
        
        plt.title("3D Human Model")
        plt.tight_layout()
        plt.show()

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