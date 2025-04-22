import numpy as np
from utils.skeleton_visualizer import SkeletonVisualizer
from utils.simple_3d_viewer import Simple3DViewer
import cv2

def get_sample_joints():
    return {
        "head": [0.5, 0.1, 0.5],
        "neck": [0.5, 0.2, 0.5],
        "left_shoulder": [0.4, 0.25, 0.5],
        "right_shoulder": [0.6, 0.25, 0.5],
        # ... остальные точки
    }

def main():
    joints = get_sample_joints()
    
    # 2D визуализация
    visualizer = SkeletonVisualizer()
    img = visualizer.draw_2d_skeleton(joints)
    cv2.imwrite("output/2d_skeleton.jpg", img)
    cv2.imshow("2D Skeleton", img)
    cv2.waitKey(2000)  # Показать на 2 секунды
    
    # 3D визуализация
    viewer = Simple3DViewer()
    viewer.plot_skeleton(joints)

if __name__ == "__main__":
    main()