import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def train_model():
    DATA_DIR = 'dataset'
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"Найдено классов: {num_classes} ({class_names})")

    targets = torch.tensor(full_dataset.targets)
    class_sample_count = torch.tensor([(targets == t).sum() for t in range(num_classes)])
    weight = 1. / class_sample_count.float()
    samples_weights = torch.tensor([weight[t] for t in targets])

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    for param in model.fc.parameters():
        param.requires_grad = True
    
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print("Начинаем обучение v3")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(full_dataset)
        epoch_acc = running_corrects.double() / len(full_dataset)

        print(f'Эпоха {epoch+1}/{EPOCHS} | Ошибка: {epoch_loss:.4f} | Точность: {epoch_acc:.4f}')

    torch.save(model.state_dict(), "art_model_v3.pth")
    print("Модель сохранена как art_model_v3.pth")

if __name__ == '__main__':
    train_model()