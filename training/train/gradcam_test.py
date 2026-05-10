import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from torchvision import models, transforms
import numpy as np
from PIL import Image
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        logit = self.model(input_tensor)
        pred_class = logit.argmax(dim=1).item()

        self.model.zero_grad()
        logit[:, pred_class].backward()

        gradients = self.gradients.cpu().numpy()[0]      # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))        # (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, pred_class


def get_model_and_layer(model_path, device):
    # 1. TẠO MODEL ĐÚNG CẤU TRÚC NHƯ KHI TRAIN
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze hết
    for param in model.parameters():
        param.requires_grad = False

    # Thay fc đúng như khi train
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    # 2. LOAD TRỌNG SỐ
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 3. BẬT GRADIENT CHO layer4 ĐỂ CÓ GRADIENT (rất quan trọng!)
    for param in model.layer4.parameters():
        param.requires_grad = True

    model = model.to(device)
    model.eval()
    return model, model.layer4[-1]  # trả về cả model và layer cuối của layer4


def apply_gradcam(model, target_layer, image_path, save_path, device, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ĐỌC ẢNH BẰNG PIL (đúng chuẩn torchvision)
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    gradcam = GradCAM(model, target_layer)
    heatmap, pred_idx = gradcam.generate(input_tensor)

    # Overlay
    img_display = np.array(pil_img.resize((224, 224)))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_display, 0.6, heatmap_color, 0.4, 0)

    # Ghi nhãn
    cv2.putText(superimposed, class_names[pred_idx], (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    cv2.imwrite(save_path, superimposed)
    print(f"Saved: {save_path} → {class_names[pred_idx]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"C:\Users\MY GEAR\DL\DL\total\modelsv_focalloss\best_model.pth"
    input_folder = r"C:\Users\MY GEAR\DL\DL\dataset\test_real_img"
    output_folder = r"C:\Users\MY GEAR\DL\DL\dataset\test_real_img\gradcam_results_fixed_v3"
    os.makedirs(output_folder, exist_ok=True)

    class_names = ["Chemical", "Real"]

    print("Loading model + enabling grad for layer4...")
    model, target_layer = get_model_and_layer(model_path, device)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:10]

    print(f"\nRunning Grad-CAM on {len(images)} images...\n")
    for i, img_name in enumerate(images):
        inp = os.path.join(input_folder, img_name)
        out = os.path.join(output_folder, f"gradcam_{i+1:02d}_{img_name}")
        apply_gradcam(model, target_layer, inp, out, device, class_names)

    print(f"\nHOÀN TẤT! Kết quả lưu tại: {output_folder}")


if __name__ == "__main__":
    main()