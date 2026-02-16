import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
import os
import json
from sklearn.model_selection import train_test_split

def train_model():
    DATASET_REPO = "dvprokofiev/russian-art" 
    MODEL_SAVE_NAME = "art_model_v3.pth"
    CLASSES_SAVE_NAME = "classes.json"
    
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.0001
    PATIENCE = 7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Используемое устройство: {DEVICE}")

    print(f"Загрузка датасета {DATASET_REPO}...")
    full_dataset = load_dataset(DATASET_REPO, split="train")

    class_names = full_dataset.features["label"].names
    num_classes = len(class_names)
    print(f"Найдено классов: {num_classes} ({class_names})")

    with open(CLASSES_SAVE_NAME, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.15, stratify=full_dataset["label"], random_state=42)
    
    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)

    targets = torch.tensor(train_dataset["label"])
    class_sample_count = torch.tensor([(targets == t).sum() for t in range(num_classes)])
    weight = 1. / class_sample_count.float()
    samples_weights = torch.tensor([weight[t] for t in targets])

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def apply_train_transforms(examples):
        examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
        return examples

    def apply_val_transforms(examples):
        examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
        return examples

    train_dataset.set_transform(apply_train_transforms)
    val_dataset.set_transform(apply_val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])
        return pixel_values, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_val_acc = 0.0
    epochs_no_improve = 0

    print("Начинаем обучение...")
    for epoch in range(EPOCHS):
        if epoch == 15:
            print(">>> Разморозка глубоких слоев для fine-tuning...")
            for param in model.features[4:].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE/10)

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

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(DEVICE), v_labels.to(DEVICE)
                v_outputs = model(v_inputs)
                _, v_preds = torch.max(v_outputs, 1)
                val_corrects += torch.sum(v_preds == v_labels.data)
        
        val_acc = val_corrects.double() / len(val_dataset)

        print(f'Эпоха {epoch+1}/{EPOCHS} | Ошибка: {epoch_loss:.4f} | Точность: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE and epoch > 20:
                print(f"Early stopping на эпохе {epoch+1}")
                break

    print(f"Обучение завершено. Лучшая точность: {best_val_acc:.4f}")
    print(f"Модель сохранена как {MODEL_SAVE_NAME}")

if __name__ == '__main__':
    train_model()