# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader
from models import ResNet18

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((32, 32)),  # Resize for CIFAR-based ResNet
    transforms.ToTensor()
    # transforms.Normalize((0.5,), (0.5,))  # Normalize (adjust if using RGB)
])

# train_dataset = "Preprocessed_animal_data/train"
train_dataset = ImageFolder(root="Dataset/train", transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Adjust epochs as needed
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

print("Training complete!")

