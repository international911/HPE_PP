import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class PoseDataset(Dataset):
    def __init__(self, csv_paths):
        self.data = []
        self.labels = []
        self.encoder = LabelEncoder()
        
        # Загружаем данные из всех CSV файлов
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            class_name = os.path.basename(os.path.dirname(csv_path))
            
            # Группируем по изображениям
            grouped = df.groupby('image_path')
            for img_path, group in grouped:
                keypoints = {}
                for _, row in group.iterrows():
                    keypoints[row['landmark']] = [row['x_norm'], row['y_norm']]
                
                # Сортируем ключевые точки по имени и объединяем в вектор
                sorted_keys = sorted(keypoints.keys())
                features = []
                for key in sorted_keys:
                    features.extend(keypoints[key])
                
                self.data.append(features)
                self.labels.append(class_name)
        
        # Кодируем метки классов
        self.labels = self.encoder.fit_transform(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label

class PoseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PoseClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

def train_model(csv_paths, model_save_path='models/pose_classifier.pth'):
    # Создаем dataset и dataloaders
    dataset = PoseDataset(csv_paths)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Инициализируем модель
    input_size = len(dataset[0][0])
    num_classes = len(dataset.encoder.classes_)
    model = PoseClassifier(input_size, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    for epoch in range(20):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
    
    # Сохраняем модель
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_classes': dataset.encoder.classes_
    }, model_save_path)
    
    return model