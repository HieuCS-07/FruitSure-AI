# 🍎🍊 FruitSure-AI: Deep Learning for Fruit Quality Inspection

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge\&logo=python\&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge\&logo=PyTorch\&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge\&logo=flask\&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge\&logo=opencv\&logoColor=white)

# 📌 Introduction

**FruitSure-AI** is a deep learning-powered fruit quality inspection system developed to classify fruits into two categories:

* **Natural Fruits**
* **Chemical-treated Fruits** (e.g. artificial wax coating or preservation chemicals)

The project combines Computer Vision and web deployment to build a real-world AI application for automated food quality inspection and consumer safety support.

---

# 🚀 Features

* Fruit classification using deep learning
* Binary classification:

  * Natural
  * Chemical-treated
* Web-based prediction system using Flask
* Multiple model experiments:

  * MobileNet
  * ResNet
* Transfer Learning & Fine-tuning
* Model evaluation with confusion matrix and metrics visualization
* Real-time image prediction interface

---

# 🧠 AI Models

The project experiments with multiple CNN architectures:

| Model     | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| MobileNet | Lightweight CNN optimized for fast inference                      |
| ResNet    | Deep residual learning architecture for robust feature extraction |

The final system uses Transfer Learning and Fine-tuning techniques to improve classification performance on custom fruit datasets.

---

# 📂 Project Structure

```plaintext
FruitSure-AI/
│
├── app/
│   └── run_app/
│       ├── static/
│       ├── templates/
│       ├── StartAPP.py
│       ├── app_mobilenet.py
│       └── app_resnet.py
│
├── models/
│   ├── models_mobilenet/
│   ├── models_resnet/
│   ├── modelsv_focalloss/
│   └── modelsv_no_aug/
│
├── results/
│   ├── results_mobilenet/
│   ├── resultsv_focalloss/
│   └── resultsv_no_aug/
│
├── training/
│   └── train/
│
├── .gitignore
└── README.md
```

---

# ⚙️ Technologies Used

## Programming Language

* Python

## Deep Learning

* PyTorch
* Transfer Learning
* CNN
* Fine-tuning

## Computer Vision

* OpenCV

## Web Framework

* Flask

## Data & Visualization

* NumPy
* Matplotlib
* Scikit-learn

---

# 🔄 Workflow

## 1. Data Collection

* Collect fruit image datasets
* Organize images into classification categories

---

## 2. Data Preprocessing

* Image resizing
* Normalization
* Data augmentation
* Dataset splitting:

  * Train
  * Validation
  * Test

---

## 3. Model Training

* Train CNN models using Transfer Learning
* Fine-tune pretrained architectures
* Experiment with different loss functions and augmentation strategies

---

## 4. Evaluation

Model performance is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Metrics
* Sample Predictions Visualization

---

## 5. Web Deployment

Users can:

* Upload fruit images
* Run AI prediction
* View classification results in real time

---

# 🎥 Demo

## Web Interface

<img width="1920" height="1080" alt="demo" src="https://github.com/user-attachments/assets/468994a5-8d6c-4287-bfa0-f7caed2097d3" />

<img width="1920" height="1080" alt="demo2" src="https://github.com/user-attachments/assets/06d8b589-4386-4cd6-9c70-493c291d16ff" />

---

# 📊 Results

The system achieved high classification performance with experimental accuracy reaching approximately **94%** on custom datasets.

The project also includes:

* Confusion matrix visualization
* Training history graphs
* Prediction samples
* Comparative model experiments

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/FruitSure-AI.git
cd FruitSure-AI
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
python app/run_app/StartAPP.py
```

Then open the local web server in your browser.

---

# 🔮 Future Improvements

Potential future upgrades:

* FastAPI deployment
* Docker containerization
* Real-time camera detection
* Mobile deployment
* Multi-class fruit classification
* Model optimization for edge devices

---

# 👨‍💻 Author

Hiếu Võ

AI / Machine Learning Enthusiast

* GitHub: [https://github.com/HieuCS-07](https://github.com/HieuCS-07)

---

# 📄 License

This project is intended for educational and research purposes.

