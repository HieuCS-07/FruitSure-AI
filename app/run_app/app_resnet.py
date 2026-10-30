from flask import Flask, request, render_template_string, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# --- THAY ĐỔI: CẤU HÌNH 2 LỚP ---
CLASSES = ['Fake_Chemical', 'Real']  # Sắp xếp theo alphabet
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = {
    'Real': 'Thật (tươi/thối tự nhiên)',
    'Fake_Chemical': 'Giả hóa chất (sáp/formalin)',
}
# --- KẾT THÚC THAY ĐỔI ---

device = torch.device('cpu')
model = None

def create_model():
    """
    Tạo model ResNet-50 với transfer learning
    Hàm này PHẢI KHỚP với file training
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        # --- TỰ ĐỘNG CẬP NHẬT ---
        # Sẽ tạo ra nn.Linear(512, 2) vì NUM_CLASSES = 2
        nn.Linear(512, NUM_CLASSES)
        # --- KẾT THÚC CẬP NHẬT ---
    )
    return model

def load_model():
    """Tải model 2-lớp mới nhất"""
    global model
    # --- THAY ĐỔI: Đường dẫn model V3 (2 Lớp) ---
    model_path = 'total/modelsv4_2class/best_model.pth' 
    # --- KẾT THÚC THAY ĐỔI ---
    
    if os.path.exists(model_path):
        try:
            model = create_model() # Tạo cấu trúc model 2-lớp
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval() # Chuyển sang chế độ đánh giá
            return True
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            print("Đảm bảo hàm create_model() trong app này khớp với file training.")
            return False
    return False

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nhận dạng trái cây (Model 2 Lớp)</title>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <style>
        /* CSS (Giữ nguyên) */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; background: #ffffff; min-height: 100vh; padding: 0; }
        .header { background: #ffffff; border-bottom: 1px solid #e5e5e5; padding: 16px 24px; position: sticky; top: 0; z-index: 100; }
        .header h1 { font-size: 20px; font-weight: 600; color: #202124; text-align: center; }
        .container { max-width: 720px; margin: 0 auto; padding: 32px 24px; }
        .subtitle { text-align: center; color: #5f6368; margin-bottom: 40px; font-size: 14px; line-height: 1.6; }
        .upload-area { border: 2px dashed #dadce0; border-radius: 12px; padding: 48px 24px; text-align: center; margin-bottom: 24px; cursor: pointer; transition: all 0.2s; background: #ffffff; }
        .upload-area:hover { background: #f8f9fa; border-color: #1a73e8; }
        .upload-area.dragover { background: #e8f0fe; border-color: #1a73e8; }
        .upload-icon { font-size: 56px; margin-bottom: 16px; color: #5f6368; }
        .material-icons { font-family: 'Material Icons'; font-weight: normal; font-style: normal; display: inline-block; line-height: 1; text-transform: none; letter-spacing: normal; word-wrap: normal; white-space: nowrap; direction: ltr; }
        .upload-text { color: #5f6368; font-size: 14px; margin-bottom: 8px; }
        .upload-hint { color: #80868b; font-size: 12px; }
        input[type="file"] { display: none; }
        .btn { background: #1a73e8; color: white; border: none; padding: 12px 24px; border-radius: 24px; font-size: 14px; font-weight: 500; cursor: pointer; width: 100%; margin-top: 16px; transition: all 0.2s; box-shadow: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15); }
        .btn:hover { background: #1557b0; box-shadow: 0 1px 3px 0 rgba(60,64,67,.3), 0 4px 8px 3px rgba(60,64,67,.15); }
        .btn:disabled { background: #f1f3f4; color: #80868b; cursor: not-allowed; box-shadow: none; }
        #preview { max-width: 100%; max-height: 320px; margin: 24px auto; display: none; border-radius: 12px; transition: opacity 0.3s ease; border: 1px solid #e5e5e5; }
        .result { margin-top: 32px; padding: 0; display: none; }
        .result h2 { color: #202124; margin-bottom: 24px; font-size: 18px; font-weight: 500; }
        .result-item { margin: 16px 0; padding: 20px; background: #f8f9fa; border-radius: 12px; border: 1px solid #e5e5e5; }
        .result-label { font-weight: 500; color: #5f6368; margin-bottom: 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
        .result-value { font-size: 20px; color: #202124; font-weight: 500; }
        .confidence-bar { background: #e8eaed; height: 8px; border-radius: 4px; overflow: hidden; margin: 12px 0 8px 0; }
        .confidence-fill { height: 100%; background: #1a73e8; transition: width 0.5s ease; }
        .confidence-text { font-size: 13px; color: #5f6368; font-weight: 500; }
        .detail-item { margin: 12px 0; padding: 12px 0; border-bottom: 1px solid #e8eaed; }
        .detail-item:last-child { border-bottom: none; }
        .detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .detail-name { font-size: 14px; color: #202124; }
        .detail-percent { font-size: 13px; font-weight: 500; color: #5f6368; }
        .loading { display: none; text-align: center; padding: 48px 32px; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 12px; margin-top: 24px; }
        .spinner { border: 4px solid #f1f3f4; border-top: 4px solid #1a73e8; border-right: 4px solid #34a853; border-radius: 50%; width: 64px; height: 64px; animation: spin 1s linear infinite; margin: 0 auto 24px auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .loading-text { color: #1a73e8; font-size: 16px; font-weight: 500; margin-bottom: 8px; }
        .loading-detail { color: #80868b; font-size: 13px; animation: pulse 1.5s ease-in-out infinite; }
        @keyframes pulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Nhận dạng trái cây Thật / Giả Hóa Chất</h1>
    </div>
    
    <div class="container">
        <p class="subtitle">Sử dụng Computer Vision và Deep Learning (Model 2 Lớp) để phân loại trái cây thành: Thật (Tự nhiên) hoặc Giả hóa chất (Sáp/Formalin).</p>
        
        <!-- Không cần nút /samples nữa, trừ khi bạn tạo lại chúng -->
        <!-- 
        <div style="text-align: center;">
            <a href="/samples" class="btn-samples">Xem ví dụ dự đoán mẫu</a>
        </div> 
        -->
        
        <div class="upload-area" id="uploadArea">
            <span class="material-icons upload-icon">cloud_upload</span>
            <p class="upload-text">Nhấp để chọn ảnh hoặc kéo thả vào đây</p>
            <p class="upload-hint">Hỗ trợ JPG, PNG (Max 10MB)</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <img id="preview" alt="Preview">
        
        <button class="btn" id="predictBtn" disabled>Phân loại ngay</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="loading-text">Đang phân tích ảnh...</p>
            <p class="loading-detail">Vui lòng đợi trong giây lát</p>
        </div>
        
        <div class="result" id="result">
            <h2>Kết quả phân loại</h2>
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        // Toàn bộ JS (giữ nguyên)
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        let selectedFile = null;

        uploadArea.onclick = () => fileInput.click();
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'));
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'));
        });
        
        uploadArea.addEventListener('drop', handleDrop);
        fileInput.addEventListener('change', handleFileSelect);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) { handleFile(files[0]); }
        }
        
        function handleFileSelect(e) {
            if (e.target.files.length > 0) { handleFile(e.target.files[0]); }
        }
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                console.error('Vui lòng chọn file ảnh!');
                return;
            }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                predictBtn.disabled = false;
                result.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        predictBtn.onclick = async () => {
            if (!selectedFile) return;
            
            predictBtn.disabled = true;
            preview.style.opacity = '0.3';
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const startTime = Date.now();
            const minLoadingTime = 1200; 
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                const elapsed = Date.now() - startTime;
                if (elapsed < minLoadingTime) {
                    await new Promise(resolve => setTimeout(resolve, minLoadingTime - elapsed));
                }
                if (data.error) {
                    displayError(data.error);
                } else {
                    displayResult(data);
                }
            } catch (error) {
                displayError('Lỗi kết nối: ' + error);
            } finally {
                loading.style.display = 'none';
                preview.style.opacity = '1';
                predictBtn.disabled = false;
            }
        };
        
        function displayError(errorMessage) {
            const content = document.getElementById('resultContent');
            content.innerHTML = `
                <div class="result-item" style="border-color: #d93025; background: #fce8e6;">
                    <div class="result-label" style="color: #d93025;">Lỗi</div>
                    <div class="result-value" style="color: #c5221f; font-size: 16px;">${errorMessage}</div>
                </div>
            `;
            result.style.display = 'block';
        }

        function displayResult(data) {
            const content = document.getElementById('resultContent');
            content.innerHTML = `
                <div class="result-item">
                    <div class="result-label">Kết quả dự đoán</div>
                    <div class="result-value">${data.predicted}</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">Độ tin cậy</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence}%"></div>
                    </div>
                    <div class="confidence-text">${data.confidence.toFixed(1)}%</div>
                </div>
                
                <div class="result-item">
                    <div class="result-label">Chi tiết các lớp</div>
                    ${Object.entries(data.all_probabilities).map(([cls, prob]) => `
                        <div class="detail-item">
                            <div class="detail-header">
                                <span class="detail-name">${cls}</span>
                                <span class="detail-percent">${prob.toFixed(1)}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${prob}%"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            result.style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# --- XÓA /samples VÀ CÁC ROUTE LIÊN QUAN ---
# Bạn có thể thêm lại nếu tạo ảnh sample mới cho 2 lớp

# @app.route('/samples')
# def samples():
#     ...

# @app.route('/resultsv3_2class/<filename>')
# def serve_result(filename):
#     from flask import send_from_directory
#     return send_from_directory('total/resultsv3_2class', filename)

# @app.route('/refresh-samples')
# def refresh_samples():
#     ...
# --- KẾT THÚC XÓA ---


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'})
    
    if model is None:
        return jsonify({'error': 'Model chưa được load hoặc load thất bại. Kiểm tra console.'})
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Transform ảnh (phải giống hệt val_test_transform)
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
        
        # --- THAY ĐỔI: LOGIC 2 LỚP ---
        predicted_class = CLASSES[predicted.item()] # Sẽ là 0 ('Fake_Chemical') hoặc 1 ('Real')
        confidence_score = confidence.item() * 100
        all_probs = probabilities[0].cpu().numpy() * 100
        
        display_label = CLASS_NAMES[predicted_class]
        
        return jsonify({
            'predicted': display_label,
            'confidence': float(confidence_score),
            'all_probabilities': {CLASS_NAMES[cls]: float(prob) for cls, prob in zip(CLASSES, all_probs)}
        })
        # --- KẾT THÚC THAY ĐỔI ---
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("="*70)
    print(" "*15 + "UNG DUNG NHAN DANG TRAI CAY (MODEL 2 LỚP)")
    print("="*70)
    
    if load_model():
        # --- THAY ĐỔI: Tên model ---
        print(f"\nModel da load thanh cong!")
        print("\nTruy cap tai: http://127.0.0.1:5000")
        print("\nNhan Ctrl+C de dung ung dung")
        print("="*70)
        app.run(host='127.0.0.1', port=5000, debug=False)
    else:
        print(f"\nKHONG TIM THAY MODEL!")
        print("Vui lòng huấn luyện model 2 lớp trước (chạy train_model_2Class.py)")
        # --- KẾT THÚC THAY ĐỔI ---