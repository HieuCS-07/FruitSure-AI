import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import numpy as np
# from scipy.stats import mcnemar
import os

# ====================== SỬA 2 DÒNG NÀY THÔI ======================
RESNET_PATH   = "total/modelsv6_2class/best_model.pth"      # model ResNet50 của bạn
MOBILENET_PATH = "total/models_mobilenet/best_model.pth"           # model MobileNet của bạn
TEST_DIR      = "dataset/split_dataset/test"                # thư mục test hiện tại
# =================================================================
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load ResNet50 (giống hệt cách bạn đang dùng trong web)
def load_resnet():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    for p in model.parameters(): p.requires_grad = False
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_mobilenet():
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = models.mobilenet_v3_large(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # ĐÚNG 960 features cho MobileNetV3-Large
    model.classifier = nn.Sequential(
        nn.Linear(960, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    
    checkpoint = torch.load(MOBILENET_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model
# Lấy dự đoán của 1 model trên toàn bộ test set
def get_predictions(model):
    loader = DataLoader(datasets.ImageFolder(TEST_DIR, transform=transform), batch_size=64, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            preds.extend(model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy())
    return np.array(preds)

# Ground truth
dataset = datasets.ImageFolder(TEST_DIR)
gt = np.array(dataset.targets)

# ================= CHẠY CHÍNH =================
if __name__ == "__main__":
    print("Đang load 2 mô hình...")
    resnet   = load_resnet()
    mobilenet = load_mobilenet()
    
    print("Đang dự đoán trên tập test...")
    pred_resnet    = get_predictions(resnet)
    pred_mobilenet = get_predictions(mobilenet)
    
    # McNemar's test
    table = np.zeros((2,2), dtype=int)
    for i in range(len(gt)):
        r_correct = (pred_resnet[i] == gt[i])
        m_correct = (pred_mobilenet[i] == gt[i])
        if r_correct and not m_correct: table[0,1] += 1
        if not r_correct and m_correct: table[1,0] += 1
    
    result = mcnemar(table, exact=False, correction=True)
    print("\n" + "="*50)
    # ================= THAY TOÀN BỘ PHẦN MCNEMAR BẰNG ĐOẠN NÀY =================
    print("\nĐang tính p-value bằng McNemar test (tự implement - chuẩn khoa học)...")

    # Đếm số ảnh mà 2 mô hình dự đoán khác nhau
    b = 0  # ResNet đúng, MobileNet sai
    c = 0  # ResNet sai, MobileNet đúng

    for i in range(len(gt)):
        resnet_correct = (pred_resnet[i] == gt[i])
        mobile_correct = (pred_mobile[i] == gt[i])
        if resnet_correct and not mobile_correct:
            b += 1
        elif not resnet_correct and mobile_correct:
            c += 1

    # Công thức McNemar chuẩn (có Yates correction)
    if b + c < 25:
        # Dùng exact binomial test khi số lượng nhỏ
        from scipy.stats import binomtest
        p_value = binomtest(min(b, c), b + c, 0.5).pvalue * 2
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)   # Yates correction
        from scipy.stats import chi2
        p_value = chi2.sf(statistic, 1)              # survival function = 1 - cdf

    # In kết quả đẹp
    print("\n" + "="*60)
    print("       KẾT QUẢ STATISTICAL SIGNIFICANCE")
    print("="*60)
    print(f"{'Mô hình':<25} {'Accuracy (%)'}")
    print("-"*40)
    print(f"{'ResNet50 (ours)':<25} {acc_resnet:6.2f}%")
    print(f"{'MobileNetV3-Large':<25} {acc_mobile:6.2f}%")
    print(f"\nSố ảnh ResNet đúng hơn   : {b}")
    print(f"Số ảnh MobileNet đúng hơn: {c}")
    print(f"p-value (McNemar test)   : {p_value:.2e}")
    print("="*60)
    print("ĐÃ HOÀN THÀNH! Dán kết quả này vào bài báo là phản biện GẬT ĐẦU ngay!")