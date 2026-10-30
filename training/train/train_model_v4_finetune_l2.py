"""
Script Tinh chỉnh Sâu (Fine-Tuning) - (Phiên bản V4)
- TẢI model V3 đã huấn luyện.
- "Mở băng" (unfreeze) layer4.
- Huấn luyện thêm 10 epochs với Learning Rate siêu nhỏ.
- Lưu kết quả vào /modelsv4 và /resultsv4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import os
from tqdm import tqdm
import json

# --- CẤU HÌNH GIAI ĐOẠN 2 (FINE-TUNING) ---
BATCH_SIZE = 32
NUM_EPOCHS_FINETUNE = 10 # Chỉ cần huấn luyện thêm 10 epochs
FINETUNE_LR = 1e-5       # Learning Rate SIÊU NHỎ (0.00001)
IMG_SIZE = 224

# Các đường dẫn
DATA_ROOT = r"C:\Users\MY GEAR\DL\DL\dataset\split_dataset" # <-- SỬA ĐƯỜNG DẪN NÀY NẾU CẦN
MODEL_V3_PATH = r"C:\Users\MY GEAR\DL\DL\total\modelsv3_l2\best_model.pth" # <-- Model V3 TỐT NHẤT
# --- KẾT THÚC CẤU HÌNH ---


# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Classes: 3 loại
CLASSES = ['Fake_Chemical', 'Fake_Origin', 'Real']
NUM_CLASSES = len(CLASSES)

# Data augmentation (Giữ nguyên như V3)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)) 
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- HÀM TẢI VÀ "MỞ BĂNG" MODEL ---
def get_finetune_model(model_path):
    """
    Tải model V3, sau đó "mở băng" (unfreeze) layer4.
    """
    print(f"Đang tải model V3 từ: {model_path}")
    
    # 1. Tạo lại cấu trúc model V3 (đã đóng băng)
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
    
    # 2. Tải trọng số V3 đã huấn luyện
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy model V3 tại {model_path}")
        print("Vui lòng chạy train_model_v3.py trước.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Tải trọng số V3 thành công.")

    # 3. "MỞ BĂNG" (UNFREEZE) CÁC LỚP SÂU HƠN
    # Chúng ta sẽ mở băng layer4 (lớp conv cuối cùng)
    print("Đang 'mở băng' (unfreezing) layer4 của ResNet...")
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Lớp fc (lớp cuối) cũng phải được huấn luyện
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)


# --- CÁC HÀM GIỮ NGUYÊN (train_one_epoch, validate) ---
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='Fine-Tuning')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': running_loss/len(dataloader), 'acc': 100.*correct/total})
    return running_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return running_loss/len(dataloader), accuracy*100, f1, all_labels, all_preds

# --- CÁC HÀM VẼ BIỂU ĐỒ (ĐỔI TÊN THÀNH V4) ---
def plot_confusion_matrix(y_true, y_pred, save_path='total/resultsv4_l2/confusion_matrix.png'):
    os.makedirs('total/resultsv4_l2', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix (Model V4 - FineTuned)', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Da luu confusion matrix: {save_path}")
    plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                          save_path='total/resultsv4_l2/training_history.png'):
    os.makedirs('total/resultsv4_l2', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch (Fine-Tuning)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Fine-Tuning Loss (V4)', fontsize=14, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch (Fine-Tuning)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Fine-Tuning Accuracy (V4)', fontsize=14, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Da luu training history: {save_path}")
    plt.close()

def visualize_predictions(model, data_loader, device, class_names, num_images=5):
    os.makedirs('total/resultsv4_l2', exist_ok=True)
    model.eval()
    all_images, all_labels, all_preds, all_probs = [], [], [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if len(all_images) >= num_images * 5: break
    indices = np.random.choice(len(all_images), size=min(num_images, len(all_images)), replace=False)
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 5))
    if num_images == 1: axes = [axes]
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off'); continue
        i = indices[idx]
        img = all_images[i].numpy().transpose((1, 2, 0))
        img = std * img + mean; img = np.clip(img, 0, 1)
        true_label = class_names[all_labels[i]]
        pred_label = class_names[all_preds[i]]
        confidence = all_probs[i][all_preds[i]] * 100
        ax.imshow(img); ax.axis('off')
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)
    plt.tight_layout()
    plt.savefig('total/resultsv4_l2/sample_predictions.png', dpi=300, bbox_inches='tight')
    print("Da luu sample predictions: total/resultsv4_l2/sample_predictions.png")
    plt.close()


# --- HÀM MAIN (ĐÃ SỬA ĐỔI) ---
def main_finetune():
    """Ham main cho Giai đoạn 2: Fine-Tuning"""
    print("="*60)
    print("STARTING MODEL FINE-TUNING (V4)")
    print("="*60)
    
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    VAL_DIR = os.path.join(DATA_ROOT, 'val')
    TEST_DIR = os.path.join(DATA_ROOT, 'test')

    if not os.path.exists(TRAIN_DIR):
        print(f"LỖI: Không tìm thấy thư mục 'train' tại {TRAIN_DIR}")
        return
    
    print("\nLoading datasets...")
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"  Classes: {train_dataset.classes}")
        global CLASSES
        if train_dataset.classes != CLASSES:
            print(f"CẢNH BÁO: Thứ tự lớp của ImageFolder là {train_dataset.classes}")
            CLASSES = train_dataset.classes
            print(f"Đã cập nhật thứ tự lớp thành: {CLASSES}")

    except Exception as e:
        print(f"Error loading data: {e}"); return
    
    # 1. Tải model V3 và "mở băng" layer4
    print("\nKhoi tao model V4 (từ V3)...")
    model = get_finetune_model(MODEL_V3_PATH)
    if model is None: return

    criterion = nn.CrossEntropyLoss()
    
    # 2. Tạo Optimizer MỚI
    # Lần này, chúng ta chỉ truyền các tham số đã "mở băng"
    # và sử dụng Learning Rate SIÊU NHỎ (FINETUNE_LR)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    print(f"Số lượng tham số sẽ được fine-tune: {len(params_to_update)}")
    optimizer = optim.Adam(params_to_update, lr=FINETUNE_LR)
    
    # Chúng ta có thể dùng lại scheduler hoặc không, với LR nhỏ thế này
    # thì không cần scheduler cũng được.
    
    print(f"\nBat dau fine-tuning {NUM_EPOCHS_FINETUNE} epochs...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    # Đánh giá model V3 "nguyên bản" trước khi fine-tune
    print("Đánh giá V3 (trước khi fine-tune) trên tập Val...")
    val_loss_v3, val_acc_v3, val_f1_v3, _, _ = validate(model, val_loader, criterion)
    best_val_acc = val_acc_v3
    print(f"  V3 Val Loss: {val_loss_v3:.4f} | V3 Val Acc: {val_acc_v3:.2f}%")

    
    for epoch in range(NUM_EPOCHS_FINETUNE):
        print(f"\n{'='*60}")
        print(f"Epoch Fine-Tune {epoch+1}/{NUM_EPOCHS_FINETUNE}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion)
        # scheduler.step() # Không cần thiết với LR nhỏ
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nKet qua Epoch {epoch+1}:")
        print(f" L_Rate: {optimizer.param_groups[0]['lr']:.7f}")
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('total/modelsv4_l2', exist_ok=True)
            torch.save(model.state_dict(), 'total/modelsv4_l2/best_model.pth')
            print(f"   ==> Saved best model V4! (Val Acc: {val_acc:.2f}%)")
    
    print("\nVe bieu do training history (V4)...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print("\nDanh gia tren test set (sử dụng model V4 tốt nhất)...")
    model.load_state_dict(torch.load('total/modelsv4_l2/best_model.pth')) # <-- V4
    test_loss, test_acc, test_f1, test_labels, test_preds = validate(model, test_loader, criterion)
    
    print("\n" + "="*60)
    print("KET QUA CUOI CUNG TREN TEST SET (V4 - FineTuned)")
    print("="*60)
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Loss: {test_loss:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))
    
    print("\nVe confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds)
    
    metrics = {
        'test_accuracy': float(test_acc),
        'test_f1_score': float(test_f1),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(best_val_acc),
        'num_epochs_finetune': NUM_EPOCHS_FINETUNE,
        'classes': CLASSES
    }
    
    with open('total/resultsv4_l2/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nDa luu metrics: total/resultsv4_l2/metrics.json")
    
    print("\nVisualizing random predictions (V4)...")
    visualize_predictions(model, test_loader, device, CLASSES, num_images=5)
    
    print("\n" + "="*60)
    print("HOAN THANH (V4)!")
    print("="*60)
    print("\nCac file ket qua:")
    print("  - modelsv4/best_model.pth")
    print("  - resultsv4/confusion_matrix.png")
    print("  - resultsv4/training_history.png")
    print("  - resultsv4/sample_predictions.png")
    print("  - resultsv4/metrics.json")

if __name__ == '__main__':
    main_finetune()