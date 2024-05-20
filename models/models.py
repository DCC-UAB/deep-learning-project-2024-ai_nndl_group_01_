import torch
import torch.nn as nn
import torch.nn.functional as F

# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
# Phocnet implementation from GH Repo
class PHOCNet(nn.Module):
    def __init__(self, num_classes=604):  # Based on  PHOC
        super(PHOCNet, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Asume entrada de imagen en escala de grises
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Capas de pooling
        self.pool1 = nn.MaxPool2d(2, 2)  # Pooling con stride de 2
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Spatial Pyramid Pooling si es necesario o capa de adaptación
        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)
        
        # Capas totalmente conectadas
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        
        # Adaptación de la capa de pooling para manejar entradas de tamaño variable
        x = self.adaptive_pool(x)
        
        # Aplanar las características para la capa totalmente conectada
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid para salida multilabel
        x = torch.sigmoid(x)
        return x