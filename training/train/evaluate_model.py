"""
Script CHỈ ĐÁNH GIÁ model đã huấn luyện trên Test Set.
Không training.
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

# --- Tải các hàm và biến cần thiết từ script training ---

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cấu hình
BATCH_SIZE = 32
IMG_SIZE = 224
CLASSES = ['Fake_Chemical', 'Fake_Origin', 'Real']  # Sắp xếp theo alphabet
NUM_CLASSES = len(CLASSES)

# Transform cho Test set
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Hàm khởi tạo model (Phải giống hệt lúc train)
def create_model():
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
    return model.to(device)

# Hàm validate
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating Test Set'):
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

# Các hàm vẽ biểu đồ
def plot_confusion_matrix(y_true, y_pred, save_path='resultsv2/confusion_matrix.png'):
    os.makedirs('resultsv2', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Da luu confusion matrix: {save_path}")
    plt.close()

def visualize_predictions(model, data_loader, device, class_names, num_images=5):
    os.makedirs('resultsv2', exist_ok=True)
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
            
            if len(all_images) >= num_images * 5: # Lấy thêm để random
                break
    
    indices = np.random.choice(len(all_images), size=min(num_images, len(all_images)), replace=False)
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 5))
    if num_images == 1: axes = [axes]
    
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off'); continue
            
        i = indices[idx]
        img = all_images[i].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        true_label = class_names[all_labels[i]]
        pred_label = class_names[all_preds[i]]
        confidence = all_probs[i][all_preds[i]] * 100
        
        ax.imshow(img)
        ax.axis('off')
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)
    
    plt.tight_layout()
    plt.savefig('resultsv2/sample_predictions_TEST.png', dpi=300, bbox_inches='tight')
    print("Da luu sample predictions: resultsv2/sample_predictions_TEST.png")
    plt.close()

# --- HÀM MAIN CHÍNH ĐỂ ĐÁNH GIÁ ---
def main_evaluate():
    """
    Hàm main chỉ để đánh giá (evaluation)
    """
    print("="*60)
    print("STARTING MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # !!! QUAN TRỌNG: SỬA ĐƯỜNG DẪN NÀY NẾU CẦN !!!
    DATA_ROOT = r"C:\Users\MY GEAR\DL\DL\dataset\split_dataset" # <-- Đường dẫn data
    TEST_DIR = os.path.join(DATA_ROOT, 'test')
    MODEL_PATH = 'modelsv2/best_model.pth' # <-- Đường dẫn model

    # Check data directory
    if not os.path.exists(TEST_DIR):
        print(f"No data found in {TEST_DIR} directory")
        return
        
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"No model file found at {MODEL_PATH}")
        print("Vui lòng kiểm tra lại đường dẫn model.")
        return

    # Load data Test
    print("\nLoading TEST dataset...")
    try:
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"  Test:  {len(test_dataset)} images | {len(test_loader)} batches")
        print(f"  Classes: {test_dataset.classes}")

        # Cập nhật CLASSES global để đảm bảo khớp
        global CLASSES
        if test_dataset.classes != CLASSES:
            print(f"CẢNH BÁO: Thứ tự lớp của ImageFolder là {test_dataset.classes}")
            print(f"Đang cập nhật CLASSES để khớp...")
            CLASSES = test_dataset.classes
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Tao model và tải weights
    print(f"\nKhoi tao model ResNet-50 va tai weights tu: {MODEL_PATH}")
    model = create_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    criterion = nn.CrossEntropyLoss()
    
    # Bắt đầu đánh giá
    print("\nDanh gia tren test set...")
    test_loss, test_acc, test_f1, test_labels, test_preds = validate(model, test_loader, criterion)
    
    print("\n" + "="*60)
    print("KET QUA DANH GIA TREN TEST SET")
    print("="*60)
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Loss: {test_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))
    
    # Confusion matrix
    print("\nVe confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds)
    
    # Luu metrics (chỉ test metrics)
    test_metrics = {
        'test_accuracy': float(test_acc),
        'test_f1_score': float(test_f1),
        'test_loss': float(test_loss),
        'classes': CLASSES
    }
    
    os.makedirs('resultsv2', exist_ok=True)
    with open('resultsv2/test_evaluation_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"\nDa luu test metrics: resultsv2/test_evaluation_metrics.json")
    
    # Visualize random predictions
    print("\nVisualizing random predictions...")
    visualize_predictions(model, test_loader, device, CLASSES, num_images=5)
    
    print("\n" + "="*60)
    print("HOAN THANH DANH GIA!")
    print("="*60)

if __name__ == '__main__':
    main_evaluate()
