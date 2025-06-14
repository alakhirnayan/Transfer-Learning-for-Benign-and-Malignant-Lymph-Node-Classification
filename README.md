Absolutely — here's your revised `README.md` file with the **segmentation publication information removed** as requested, while retaining all other relevant content:

---

````markdown
# Mediastinal Lymph Node Classification Using Deep Learning

This repository supports the IEEE Access publication focused on **mediastinal lymph node classification** using **deep learning** and **feature engineering techniques**. It includes preprocessing scripts, feature extraction tools, and training notebooks for multiple CNN-based classifiers.

---

## 📘 Publication

**Classification**  
_"Navigating Malignancy: Deep Learning for Mediastinal Lymph Node Classification"_  
IEEE Access, 2024. [DOI: 10.1109/ACCESS.2024.3491414](https://doi.org/10.1109/ACCESS.2024.3491414)

---

## 🎯 Objectives

- **Classification**: Identify benign vs malignant lymph nodes using VGG16, VGG19, ResNet50, InceptionV3, and Xception models.
- **Preprocessing**: Convert raw DICOM files, apply LBP/ORB/FAST feature extraction to enrich datasets.

---

## 🗃️ Repository Structure

```bash
├── convert1.py / convert2.py          # DICOM to JPG conversion (Malignant / Benign)
├── LBP1.py / LBP2.py                  # Local Binary Pattern extraction
├── ORB1.py / ORB2.py                  # ORB keypoint visualization
├── FAST1.py / FAST2.py                # FAST feature detection
├── Split_Data.ipynb                   # Train/test dataset splitting
├── testing_load_image.ipynb           # Utility to visualize and test images
├── VGG16_image_Classification.ipynb   # Deep learning classifier (best performance)
├── VGG19_image_Classification.ipynb
├── resnet50_image_classifier.ipynb
├── Xception_image_Classification.ipynb
├── InceptionV3_image_Classification.ipynb
````

---

## 🛠️ Environment Setup (Python 3.8 using Miniconda)

### 1. Create and Activate Environment

```bash
conda create -n lymph_node_env python=3.8
conda activate lymph_node_env
```

### 2. Install Required Libraries

```bash
conda install -c conda-forge opencv scikit-image matplotlib imageio jupyterlab numpy pandas pillow
pip install tensorflow keras pydicom
```

> Ensure your GPU drivers and CUDA toolkit are installed if training on GPU.

---

## ▶️ How to Run the Code

### 🔹 Step 1: Prepare the Dataset

Organize your DICOM files into:

```
/data/Lymph_Data/Data/Malignant/...
/data/Lymph_Data/Data/Benign/...
```

### 🔹 Step 2: Convert DICOM to JPG

```bash
python convert1.py     # For malignant
python convert2.py     # For benign
```

### 🔹 Step 3: Feature Extraction

```bash
python LBP1.py         # LBP for benign
python LBP2.py         # LBP for malignant

python ORB1.py         # ORB for benign
python ORB2.py         # ORB for malignant

python FAST1.py        # FAST for benign
python FAST2.py        # FAST for malignant
```

### 🔹 Step 4: Split the Data

Open and run `Split_Data.ipynb` to split the dataset into training and testing folders.

### 🔹 Step 5: Train Classifiers

Use any of the provided Jupyter notebooks to train and evaluate models:

* `VGG16_image_Classification.ipynb` (Recommended - best accuracy)
* `VGG19_image_Classification.ipynb`
* `resnet50_image_classifier.ipynb`
* `Xception_image_Classification.ipynb`
* `InceptionV3_image_Classification.ipynb`

> Each notebook loads data, preprocesses it, builds a model, trains, and evaluates performance.

### 🔹 Step 6: Test & Visualize

Use `testing_load_image.ipynb` to load sample images and validate your preprocessing outputs (e.g., LBP or keypoints).

---

## 📈 Results Summary

* **Classification Accuracy (VGG16)**: 98.08%
* Detailed metrics (Precision, Recall, etc.) are reported in the publication.

---

## 🧑‍💻 Contributor

**A. A. Nayan**
For academic use, please cite the IEEE publication linked above.

---

## 📄 License

License – free to use, distribute, and modify with proper citation.

```

