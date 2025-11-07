# 🐾 Animal Faces Image Classification

A deep learning project built using **PyTorch** to classify animal images (Cats, Dogs, and Wild Animals) from the **Animal Faces dataset (AFHQ)**.  
This project demonstrates image preprocessing, data augmentation, CNN design from scratch, and performance evaluation with visualization.

---

## 📘 Overview

The goal of this project is to develop a **Convolutional Neural Network (CNN)** capable of accurately classifying animal faces into 3 categories.  
The model was trained on the **AFHQ dataset** from Kaggle and achieved **96.53% test accuracy** after 20 epochs of training.

---

## 📊 Dataset

**Dataset:** [Animal Faces Dataset (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces)  
**Classes:** 3 (Cats, Dogs, Wild Animals)  
**Image Size:** 256×256  
**Split:**
- 70% Training  
- 15% Validation  
- 15% Testing  

---

## ⚙️ Tech Stack & Libraries

- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:**  
  `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `PIL`, `sklearn`

---
## 🔄 Data Preprocessing & Augmentation

The following transformations were applied using `torchvision.transforms`:
- Resize images to **256×256**
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Conversion to tensor and normalization  

These augmentations help improve the model’s generalization ability and reduce overfitting.

---

## 🧠 Model Architecture

A **CNNModel** built from scratch with:
- 4 Convolutional layers (with ReLU activation & MaxPooling)
- Fully Connected (Dense) layers for classification
- Output 3-class prediction

**Optimizer:** Adam  
**Loss Function:** CrossEntropyLoss  
**Epochs:** 20  
**Batch Size:** 32  

---

## 📈 Results

| Metric | Training | Validation | Testing |
|:-------:|:----------:|:------------:|:-----------:|
| **Accuracy** | 98.93% | 96.86% | 96.53% |
| **Loss** | 0.216 | 0.146 | 0.0167 |

- The model achieved **96.53% accuracy** on unseen test data.  
- Training and validation curves showed strong convergence and no major signs of overfitting.

---

## 📉 Training Visualization

The notebook includes visualizations for:
- Training vs Validation **Loss**
- Training vs Validation **Accuracy**

These plots demonstrate smooth convergence and stable learning across epochs.

--
