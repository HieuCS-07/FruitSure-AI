# 🍎🍊 FruitSure-AI: Deep Learning for Fruit Quality Inspection

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

## 📌 Introduction
**FruitSure-AI** is a deep learning-based web application designed to classify fruits into two primary categories: **Natural** and **Chemical-treated** (e.g., artificial wax coatings). This project aims to assist in automated food quality inspection, ensuring consumer safety and product transparency.

## 🚀 Key Features
*   **High Accuracy:** Achieved an operational accuracy of over **94%** in resolving subtle visual ambiguities on fruit surfaces.
*   **End-to-End Pipeline:** From automated data cleaning to eliminate noise/label inconsistencies to full-stack web deployment.
*   **Advanced AI Model:** Utilized Transfer Learning and Fine-tuning on the **ResNet-50** architecture for robust feature extraction.
*   **User-Friendly Web UI:** A simple Flask-based interface allowing users to upload fruit images and get instant predictions.

## 🎥 Demo
<img width="1920" height="1080" alt="demo" src="https://github.com/user-attachments/assets/468994a5-8d6c-4287-bfa0-f7caed2097d3" />

<img width="1920" height="1080" alt="demo2" src="https://github.com/user-attachments/assets/06d8b589-4386-4cd6-9c70-493c291d16ff" />


## 📂 Project Structure
```text
FruitSure-AI/
│
├── src/
│   ├── run_app/                 # Backend Web App (Flask)
│   │   ├── templates/           # HTML files (index.html, about.html)
│   │   └── StartAPP.py          # Main Flask application logic
│   │
│   └── train/                   # Model Training Scripts
│       └── train_model_2Class.py # Data pipeline and ResNet-50 training logic
│
└── README.md
