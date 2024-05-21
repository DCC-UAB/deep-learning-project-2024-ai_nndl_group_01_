import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import wandb

from models import PHOCNet
from data_loader import MJSynthDataset


# Inicializar WandB
wandb.init(project="phocnet-training")

# Definir hiperparámetros y configuraciones
config = wandb.config
config.epochs = 20
config.batch_size = 64
config.learning_rate = 0.001

# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Preparar los datos
train_dataset = MJSynthDataset(
    root_dir='/path/to/90kDICT32px',
    annotation_file='/path/to/annotation_train.txt',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

val_dataset = MJSynthDataset(
    root_dir='/path/to/90kDICT32px',
    annotation_file='/path/to/annotation_val.txt',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Inicializar el modelo, el optimizador y la función de pérdida
model = PHOCNet(n_out=604)  # Cambia n_out según tu tarea
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.BCELoss()

# Registrar el modelo en WandB
wandb.watch(model, log="all")

# Bucle de entrenamiento
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}')
    
    # Registrar los valores de pérdida en WandB
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    wandb.log({"val_loss": avg_val_loss})