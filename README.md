# Vasudha-AI
# 🌍 Deep Learning Optimization in Earth Observation for Imbalanced Datasets

This project was built for the ISRO Hackathon 2025, focusing on improving the detection of **rare events** (e.g., water bodies, landslides) in **Earth Observation (EO)** satellite imagery using **deep learning** techniques.

Our goal is to handle the **class imbalance problem** commonly found in EO data using a combination of:
- Advanced data augmentation,
- Balanced loss functions,
- Attention mechanisms,
- Synthetic oversampling strategies.

---

## 🧠 Problem Overview

Standard deep learning models underperform when classes are imbalanced — which is typical in EO imagery. This affects detection of rare classes like small water bodies or deforestation patches. Our solution integrates:

- **Synthetic oversampling** for rare classes  
- **Focal and Dice loss** to focus learning  
- **Attention-enhanced architectures** for better feature extraction

---

## 🧪 Our Solution Pipeline

Raw Satellite Imagery (LISS-IV)
↓
Preprocessing & Augmentation (Role A)
↓
Modeling & Loss Design (Role B)
↓
Training & Evaluation with Metrics (Role C)
↓
Visual Reports & Web Demo Interface (Role D)

## 🧑‍💻 Roles & Responsibilities

| Role | Task |
|------|------|
| A | Preprocessing, augmentation, patching |
| B | Model selection (U-Net, DeepLabv3+), custom loss |
| C | Training loop, metrics (IoU, F1, Recall), confusion matrix |
| D | Innovation strategy, reporting, web demo (Streamlit) |

---

## 🗂️ Repository Structure
vasudha-ai/
│
├── data/ # Input from Role A
│ ├── images/ # Satellite tiles
│ ├── masks/ # Segmentation masks
│ └── split/ # train.txt / val.txt / test.txt
│
├── models/ # Input from Role B
│ ├── unet.py # Model architecture
│ └── losses.py # DiceLoss, FocalLoss, etc.
│
├── training_evaluation_pipeline/ # Role C
│ ├── train.py # Training script
│ ├── metrics.py # IoU, F1, Recall
│ ├── utils.py # Data loading, plots
│ ├── config.py # Paths, hyperparameters
│ └── runs/ # Logs, saved models, results
│
├── demo_app/ # Role D
│ └── streamlit_app.py # Streamlit web demo
│
├── requirements.txt # Required Python packages
└── README.md # Project summary

---

## 📊 Metrics We Track

- **IoU (Intersection over Union)** – Per-class and mean  
- **F1-Score** – Focus on rare classes  
- **Recall** – For sensitivity on underrepresented regions  
- **Confusion Matrix** – For visual inspection of class confusion

---

## 🌐 Streamlit Web Demo

Run the live demo locally to see segmentation results:

🔍 Key Innovations
🎯 Loss Fusion Strategy – Focal + Dice Loss
🧬 Synthetic Copy-Paste Oversampling for rare classes
🔍 Patch-Aware Training to ensure rare class visibility
🧠 Attention Layers in segmentation heads

