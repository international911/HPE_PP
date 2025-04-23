import csv
import numpy as np

def load_keypoints(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return {row['joint']: [float(row['x']), float(row['y']), float(row.get('z', 0))] 
               for row in reader}