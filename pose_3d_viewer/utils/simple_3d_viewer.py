import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Simple3DViewer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
    
    def plot_skeleton(self, joints):
        connections = [
            ("head", "neck"),
            ("neck", "left_shoulder"),
            # ... остальные соединения из config
        ]
        
        for joint1, joint2 in connections:
            if joint1 in joints and joint2 in joints:
                x = [joints[joint1][0], joints[joint2][0]]
                y = [joints[joint1][1], joints[joint2][1]]
                z = [joints[joint1][2] if len(joints[joint1])>2 else 0, 
                     joints[joint2][2] if len(joints[joint2])>2 else 0]
                self.ax.plot(x, y, z, 'b-', linewidth=2)
        
        plt.title("3D Skeleton Visualization")
        plt.tight_layout()
        plt.show()