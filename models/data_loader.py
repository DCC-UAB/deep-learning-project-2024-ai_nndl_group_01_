import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MJSynthDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (string): Directorio con todas las imágenes.
            annotation_file (string): Archivo de anotaciones.
            transform (callable, optional): Transformaciones a aplicar en las imágenes.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Leer el archivo de anotaciones
        self.image_labels = []
        with open(annotation_file, 'r') as file:
            for line in file:
                img_name, label = line.strip().split()
                self.image_labels.append((img_name, label))
                    
    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convertir la etiqueta a un tensor de caracteres
        label_tensor = torch.tensor([ord(char) for char in label], dtype=torch.long)
        
        return image, label_tensor

# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 32)),  # Cambia el tamaño según tus necesidades
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Crear el dataset y el dataloader
train_dataset = MJSynthDataset(
    root_dir='/path/to/90kDICT32px',
    annotation_file='/path/to/annotation_train.txt',  # Cambia esto al archivo correcto
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Similar para validación y test
val_dataset = MJSynthDataset(
    root_dir='/path/to/90kDICT32px',
    annotation_file='/path/to/annotation_val.txt',  # Cambia esto al archivo correcto
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = MJSynthDataset(
    root_dir='/path/to/90kDICT32px',
    annotation_file='/path/to/annotation_test.txt',  # Cambia esto al archivo correcto
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
