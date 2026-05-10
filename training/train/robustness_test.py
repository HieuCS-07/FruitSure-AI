import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm
import random
from PIL import Image, ImageFilter, ImageEnhance
import os

# ================= CẤU HÌNH (GIỐNG HỆT WEB CỦA BẠN) =================
TEST_DIR = "dataset/split_dataset/test"  # Thư mục test của bạn
MODEL_PATH = "total/modelsv6_2class/best_model.pth"  # ĐÚNG ĐƯỜNG DẪN BẠN ĐANG DÙNG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 2
CLASSES = ['Chemical', 'Real']

# def create_model():
#     weights = models.ResNet50_Weights.IMAGENET1K_V2
#     model = models.resnet50(weights=weights)
    
#     for param in model.parameters():
#         param.requires_grad = False

#     num_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 512),
#         nn.ReLU(),
#         nn.Dropout(0.3),
#         nn.Linear(512, NUM_CLASSES)
#     )
    
#     return model

# def load_model():
#     print(f"Đang load model từ: {MODEL_PATH}")
#     if not os.path.exists(MODEL_PATH):
#         print("Không tìm thấy file model!")
#         exit()
    
#     model = create_model()
#     try:
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#         print("Load model thành công!")
#     except Exception as e:
#         print(f"Lỗi load model: {e}")
#         exit()
    
#     model.to(DEVICE)
#     model.eval()
#     return model
def create_model():
    """Tạo model ResNet-50 với transfer learning"""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES) 
    )
    
    return model.to(DEVICE)

def load_model():
    """Tải model và TRẢ VỀ model (không trả True/False)"""
    model_path = 'total/modelsv4_2class/best_model.pth'  # hoặc v4, tùy bạn đang dùng cái nào đúng
    
    if not os.path.exists(model_path):
        print(f"KHÔNG TÌM THẤY FILE MODEL TẠI: {model_path}")
        exit()
    
    print(f"Đang tải model từ: {model_path}")
    try:
        temp_model = create_model()  # Tạo đúng cấu trúc 2 lớp
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        
        # Xử lý trường hợp save cả dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            
        temp_model.load_state_dict(checkpoint)
        print("LOAD MODEL THÀNH CÔNG 100%!")
        temp_model.to(DEVICE)
        temp_model.eval()
        return temp_model  # TRẢ VỀ MODEL THẬT
        
    except Exception as e:
        print(f"Lỗi load model: {e}")
        print("Gợi ý: thử đổi đường dẫn model_path hoặc dùng weights_only=False")
        exit()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def apply_noise(img_tensor, noise_type):
    img = transforms.ToPILImage()(img_tensor.clamp(0, 1))
    if noise_type == "gaussian":
        noise = torch.randn_like(img_tensor) * 0.15
        return torch.clamp(img_tensor + noise, 0, 1)
    elif noise_type == "motion_blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
    elif noise_type == "brightness":
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Brightness(img).enhance(factor)
    elif noise_type == "contrast":
        img = ImageEnhance.Contrast(img).enhance(0.5)
    elif noise_type == "jpeg":
        img.save("tmp.jpg", quality=30)
        img = Image.open("tmp.jpg")
    return transforms.ToTensor()(img)

def evaluate(loader, model, noise_type=None):
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Testing {noise_type or 'Clean'}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if noise_type:
                images = torch.stack([apply_noise(img.cpu(), noise_type) for img in images]).to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

if __name__ == "__main__":
    model = load_model()
    dataset = ImageFolder(TEST_DIR, transform=val_transform)
    print(f"Classes detected: {dataset.classes}")  # In ra để kiểm tra thứ tự
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

    results = {}
    results["Clean"] = evaluate(loader, model)

    noises = [
        ("Gaussian Noise", "gaussian"),
        ("Motion Blur", "motion_blur"),
        ("Brightness ±40%", "brightness"),
        ("Low Contrast", "contrast"),
        ("JPEG Compression", "jpeg")
    ]

    for name, ntype in noises:
        print(f"\n→ Testing: {name}")
        results[name] = evaluate(loader, model, noise_type=ntype)

    results["Combined 3 Noises"] = np.mean([
        evaluate(loader, model, "gaussian"),
        evaluate(loader, model, "brightness"),
        evaluate(loader, model, "contrast")
    ])

    # ================= IN BẢNG ĐẸP =================
    print("\n" + "="*60)
    print("       ROBUSTNESS EVALUATION RESULTS (ĐÃ SỬA ĐÚNG MODEL)")
    print("="*60)
    for name, acc in results.items():
        delta = acc - results["Clean"]
        print(f"{name:30} : {acc:6.2f}%   (Δ {delta:+5.2f}%)")
    print("="*60)