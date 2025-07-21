import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

# === 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Set transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === 3. Load datasets
train_data = datasets.ImageFolder("datasets/bottle_materials/plastic_glass/train", transform=transform)
valid_data = datasets.ImageFolder("datasets/bottle_materials/plastic_glass/valid", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16)

# === 4. Load a pre-trained lightweight model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: plastic, glass
#model.load_state_dict(torch.load("models/bottle_material_classifier.pt"))
model = model.to(device)

#alternate model
model = models.resnet50(weights = 'IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)

# === 5. Set loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === 6. Training loop (try 10 epochs first)
for epoch in range(10):
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

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# === 7. Save the model
torch.save(model.state_dict(), "bottle_material_classifier.pt")
print("✅ Model saved as 'bottle_material_classifier.pt'")


model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Validation accuracy: {correct / total:.2%}")