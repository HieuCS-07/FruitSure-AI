from flask import Flask, request, render_template_string, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

CLASSES = ['Chemical', 'Real']  
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = {
    'Real': 'Thật (tươi/thối tự nhiên)',
    'Chemical': 'Giả hóa chất (sáp/formalin)',
}

device = torch.device('cpu')
model = None
# --- KẾT THÚC CẤU HÌNH ---

# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #
# ------ đối với model Resnet ------


# def create_model():
#     model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#     for param in model.parameters():
#         param.requires_grad = False
#     # Unfreeze layer4 + fc
#     for name, param in model.named_parameters():
#         if "layer4" in name or "fc" in name:
#             param.requires_grad = True
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
#     return model.to(device)

# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #
# ------ đối với model Resnet Unfreeze layer4 + fc------
# def create_model():
#     model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#     for param in model.parameters():
#         param.requires_grad = False
#     # Unfreeze layer4 + fc
#     for name, param in model.named_parameters():
#         if "layer4" in name or "fc" in name:
#             param.requires_grad = True
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
#     return model.to(device)

# # # ------------------------------------------------------------------------------------ #
# # ------------------------------------------------------------------------------------ #
# # ------ đối với model Resnet fc------
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

def load_model():
    global model
    model_path = "total/modelsv4_2class/best_model.pth"
    
    if os.path.exists(model_path):
        try:
            model = create_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return True
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            print("Đảm bảo hàm create_model() trong app này khớp với file training.")
            return False
    return False

# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #

# ------ đối với model MobileNetV3 ------
# def create_model():
#     weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
#     model = models.mobilenet_v3_large(weights=weights)
#     for param in model.parameters():
#         param.requires_grad = False
#     num_features = 960 
#     model.classifier = nn.Sequential(
#         nn.Linear(num_features, 512),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.3),
#         nn.Linear(512, NUM_CLASSES)
#     )
#     return model

# def load_model():
#     global model
#     model_path = 'total/models_mobilenet/best_model.pth'
#     if os.path.exists(model_path):
#         try:
#             model = create_model()
#             model.load_state_dict(torch.load(model_path, map_location=device))
#             model.to(device)
#             model.eval()
#             return True
#         except Exception as e:
#             print(f"Lỗi khi tải model: {e}")
#             return False
#     return False

# ------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------ #

# @app.route('/')
# def landing():
#     # Trả về trang giới thiệu mới
#     return render_template('landing.html') 

# @app.route('/app')
# def index():
#     # Trả về trang ứng dụng chính
#     return render_template('index.html') 

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Nhận dữ liệu JSON (chứa Base64 string)
#     data = request.get_json()
#     if not data or 'image' not in data:
#         return jsonify({'error': 'Không nhận được dữ liệu ảnh.'}), 400
    
#     try:
#         image_data = base64.b64decode(data['image'])
#         image = Image.open(io.BytesIO(image_data)).convert('RGB')
#     except Exception as e:
#         return jsonify({'error': f'Lỗi giải mã ảnh (Base64): {e}'}), 400
    
#     if model is None:
#         return jsonify({'error': 'Model chưa được load hoặc load thất bại. Kiểm tra console.'}), 500
    
#     try:
#         # Transform ảnh (phải giống hệt val_test_transform)
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
        
#         img_tensor = transform(image).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             outputs = model(img_tensor)
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
#             confidence, predicted = torch.max(probabilities, 1)
        
#         # --- LOGIC 2 LỚP ---
#         predicted_class = CLASSES[predicted.item()]
#         confidence_score = confidence.item() * 100
#         all_probs = probabilities[0].cpu().numpy() * 100
        
#         display_label = CLASS_NAMES[predicted_class]
        
#         return jsonify({
#             'predicted': display_label,
#             'confidence': float(confidence_score),
#             'all_probabilities': {CLASS_NAMES[cls]: float(prob) for cls, prob in zip(CLASSES, all_probs)}
#         })
#         # --- KẾT THÚC LOGIC ---
    
#     except Exception as e:
#         return jsonify({'error': f'Lỗi phân tích model: {e}'}), 500
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không nhận được file ảnh (Lỗi request.files)'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file ảnh.'}), 400
        
    if model is None:
        if not load_model():
             return jsonify({'error': 'Model chưa được load trên Server. Vui lòng kiểm tra log.'}), 500
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASSES[predicted.item()]
        confidence_score = confidence.item() * 100
        all_probs = probabilities[0].cpu().numpy() * 100
        
        display_label = CLASS_NAMES[predicted_class]
        
        return jsonify({
            'predicted': display_label,
            'confidence': float(confidence_score),
            'all_probabilities': {CLASS_NAMES[cls]: float(prob) for cls, prob in zip(CLASSES, all_probs)}
        })
    
    except Exception as e:
        print(f"LOI KHI DU DOAN: {e}")
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

if __name__ == '__main__':
    print("="*70)
    print(" "*15 + "UNG DUNG NHAN DANG TRAI CAY")
    print("="*70)
    
    if load_model():
        print(f"\nModel da load thanh cong!")
        print("\nTruy cap tai: http://127.0.0.1:5000")
        print("\nNhan Ctrl+C de dung ung dung")
        print("="*70)
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print(f"\nKHONG TIM THAY MODEL!")
        print("Vui lòng huấn luyện model trước")