# 🩻 Chest X-Ray Pneumonia Classifier

**Live Demo:** [https://xray-app-tapas-parida.streamlit.app/](https://xray-app-tapas-parida.streamlit.app/)

A deep learning web application that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a VGG16-based transfer learning model, deployed with Streamlit.

---

## 📌 Project Overview

This project builds an end-to-end medical image classification pipeline that detects pneumonia from chest X-ray images. Two models were developed and compared:

1. **Custom CNN (trained from scratch)** — baseline model
2. **VGG16 (Transfer Learning)** — production model used in deployment

The **VGG16 model** achieved significantly higher test accuracy and is the one served through the Streamlit web application.

---

## 🎯 Motivation

Pneumonia is a leading cause of death worldwide, especially among children and the elderly. Early detection from chest X-rays is critical but requires radiologist expertise that is often unavailable in low-resource settings. This project explores whether a deep learning model can provide fast, accessible preliminary screening support.

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It is **not** a certified medical device and must not be used as a substitute for professional radiologist diagnosis.

---

## 🚀 Live Application

Try the model on your own X-ray images:

👉 **[https://xray-app-tapas-parida.streamlit.app/](https://xray-app-tapas-parida.streamlit.app/)**

**How to use:**
1. Open the link above.
2. Upload a chest X-ray image (`.jpg`, `.jpeg`, or `.png`).
3. Click **PREDICT**.
4. The model returns the predicted class (**NORMAL** or **PNEUMONIA**) along with a confidence score.

---

## 📊 Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** ~5,800+
- **Split:**
  - Training: 5,226 images
  - Validation: 16 images
  - Test: 624 images
- **Classes:**
  - `NORMAL` → label `0`
  - `PNEUMONIA` → label `1`
- **Class Distribution:** Imbalanced (~1,341 normal vs. ~3,875 pneumonia in training)

---

## 🏗️ Model Architecture

### Model 1 — Custom CNN (Baseline)

A sequential convolutional network trained from scratch:

| Layer | Details |
|---|---|
| Conv2D (64 filters, 3×3) + ReLU + MaxPool + Dropout(0.2) | Feature extraction |
| Conv2D (128 filters, 3×3) + ReLU + MaxPool + Dropout(0.2) | Feature extraction |
| Conv2D (256 filters, 3×3) + ReLU + MaxPool + Dropout(0.2) | Feature extraction |
| Flatten | Converts 3D → 1D |
| Dense (64, ReLU) + Dropout(0.5) | Fully connected |
| Dense (1) + Sigmoid | Binary output |

- **Total parameters:** ~2.0M
- **Input shape:** `(100, 100, 1)` — grayscale

### Model 2 — VGG16 Transfer Learning (Deployed)

VGG16 pretrained on ImageNet, with the original classifier removed and a custom head attached:

| Component | Details |
|---|---|
| Base | VGG16 (`include_top=False`, `weights='imagenet'`) |
| Base Layers | **Frozen** (non-trainable) |
| Flatten | 4,608 features |
| Dense (256, ReLU) | Custom head |
| Dense (128, ReLU) | Custom head |
| Dense (64, ReLU) | Custom head |
| Dense (1, Sigmoid) | Binary output |

- **Total parameters:** ~15.9M
- **Trainable parameters:** ~1.2M (custom head only)
- **Input shape:** `(100, 100, 3)` — RGB

---

## ⚙️ Training Configuration

| Parameter | Custom CNN | VGG16 |
|---|---|---|
| Image Size | 100 × 100 | 100 × 100 |
| Channels | Grayscale (1) | RGB (3) |
| Batch Size | 4 | 32 |
| Epochs | 10 | 10 |
| Optimizer | Adam | Adam |
| Loss Function | Binary Crossentropy | Binary Crossentropy |
| Data Augmentation | ❌ None | ✅ Shear, Zoom, Horizontal Flip |
| Normalization | ÷ 255 | ÷ 255 (via ImageDataGenerator) |

---

## 📈 Results

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
|---|---|---|---|
| **Custom CNN** | 97.83% | 93.75% | **74.68%** |
| **VGG16 (Transfer Learning)** | 95.66% | 75%–93.75% | **89.26%** |

### Key Observations

- The **custom CNN overfit heavily** — training accuracy reached 97.8% while test accuracy stalled at 74.7%.
- **VGG16 generalized much better** due to:
  - Pretrained ImageNet feature extractors
  - Data augmentation (shear, zoom, horizontal flip)
  - A deeper convolutional backbone
- The validation set was **only 16 images**, making validation accuracy noisy and unreliable (a single misclassification shifts the score by 6.25%).

---

## 🧠 Key Concepts Used

- **Convolutional Neural Networks (CNNs)** — spatial feature extraction
- **Transfer Learning** — reuse of ImageNet-pretrained VGG16 weights
- **Layer Freezing** — prevent catastrophic forgetting of pretrained features
- **Data Augmentation** — reduces overfitting by generating transformed training samples
- **Dropout Regularization** — randomly disables neurons during training
- **Max Pooling** — spatial downsampling with translation invariance
- **Sigmoid Activation** — outputs probability for binary classification
- **Binary Crossentropy Loss** — standard for two-class problems
- **Feature Scaling (÷ 255)** — normalizes pixel values to `[0, 1]`
- **Google Drive Mounting** — attaches Drive as a virtual filesystem in Google Colab

---

## 🗂️ Project Structure

```
chest_xray_project/
│
├── app.py                        # Streamlit web application (deployed)
├── my_x_ray_proj.ipynb           # Jupyter/Colab notebook (training + evaluation)
├── model_vgg_tapas.h5            # Trained VGG16 model (deployed)
├── model_1_tapas.h5              # Trained custom CNN model (baseline)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Deep Learning** | TensorFlow / Keras |
| **Image Processing** | OpenCV, Pillow, `tf.keras.preprocessing.image` |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib |
| **Deployment** | Streamlit (Streamlit Community Cloud) |
| **Notebook Environment** | Google Colab |
| **Cloud Storage** | Google Drive |

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd chest_xray_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** `model_vgg_tapas.h5` must be present in the same directory as `app.py`. If the file is large (>100 MB), use Git LFS or store it on cloud storage and load it by URL.

---

## 🔮 Future Improvements

- [ ] **Fix confidence display bug** — show correct confidence for both classes (currently displays P(PNEUMONIA) regardless of prediction).
- [ ] **Larger validation set** — the current 16-image validation set is statistically unreliable.
- [ ] **Class weighting** — handle the 1:3 class imbalance.
- [ ] **Precision, Recall, F1, AUC-ROC** — accuracy alone is misleading for imbalanced medical data.
- [ ] **Fine-tuning** — unfreeze the last VGG16 block and retrain with a low learning rate.
- [ ] **Grad-CAM visualization** — highlight regions the model is focusing on (crucial for clinical trust).
- [ ] **Cross-validation** — provide more reliable performance estimates.
- [ ] **Migrate to `.keras` format** — the HDF5 format is legacy.
- [ ] **Model optimization** — quantization / ONNX export for faster inference.

---

## ⚠️ Known Limitations

1. **Overfitting in the custom CNN** — the training-vs-test accuracy gap is significant.
2. **Tiny validation set (16 images)** — validation metrics are unreliable.
3. **Confidence display bug** — confidence shows `P(PNEUMONIA)` even when the prediction is NORMAL.
4. **Class imbalance** — not addressed with class weights or resampling.
5. **No explainability** — no Grad-CAM or saliency maps, so predictions are hard to audit clinically.
6. **Accuracy-only evaluation** — precision/recall/AUC would be more informative for medical use.
7. **Horizontal flip augmentation** may be questionable for X-rays (swaps left/right lung).

---

## 📚 References

- [Chest X-Ray Images (Pneumonia) Dataset — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [VGG16 — Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [Keras Applications Documentation](https://keras.io/api/applications/vgg/)
- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👨‍💻 Author

**Tapas Parida**

- Live Demo: [https://xray-app-tapas-parida.streamlit.app/](https://xray-app-tapas-parida.streamlit.app/)
- Feel free to connect and provide feedback!

---

## 📄 License

This project is intended for **educational and research purposes only**. It is not a certified medical diagnostic tool and should not be used in clinical decision-making.

---

## 🙏 Acknowledgements

- **Kaggle** and **Paul Mooney** for the chest X-ray dataset.
- **Google Colab** for free GPU compute.
- **Streamlit Community Cloud** for free hosting.
- The **Keras** team for the pretrained VGG16 model.

---

⭐ **If you found this project helpful, consider giving it a star!**
