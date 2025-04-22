import cv2
import numpy as np
import json

class SkeletonVisualizer:
    def __init__(self, config_path="../config/skeleton_connections.json"):
        with open(config_path) as f:
            self.connections = json.load(f)["connections"]
    
    def draw_2d_skeleton(self, joints, image=None):
        if image is None:
            image = 255 * np.ones((1000, 1000, 3), dtype=np.uint8)
        
        for joint1, joint2 in self.connections:
            if joint1 in joints and joint2 in joints:
                x1, y1 = joints[joint1][:2]
                x2, y2 = joints[joint2][:2]
                h, w = image.shape[:2]
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        
        return image