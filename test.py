import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import sys
import os

def predict(url_or_path):
    MODEL_PATH = "art_model_v3.pth"
    
    # like in the dataset
    class_names = [
        'Academism',   # 0
        'Avantgard',   # 1
        'Barokko',     # 2
        'Iconopis',    # 3
        'Modern',      # 4
        'Parsuna',     # 5
        'Peredviz',    # 6
        'Romantism'    # 7
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )

    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл весов '{MODEL_PATH}' не найден в текущей директории!")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Ошибка при загрузке state_dict: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        if url_or_path.startswith('http'):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url_or_path, headers=headers, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(url_or_path).convert('RGB')
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    style = class_names[predicted_class]
    print("\n" + "═"*45)
    print(f"Результат работы")
    print("═"*45)
    print(f" Предполагаемый стиль: {style.upper()}")
    print(f" Уверенность:          {confidence.item()*100:.2f}%")
    print("═"*45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python3 test.py <ссылка_на_картину_или_путь_к_файлу>")
    else:
        predict(sys.argv[1])