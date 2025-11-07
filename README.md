# Breast Cancer Classification Using Deep Learning and Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive machine learning project for classifying breast cancer histopathological images using various deep learning architectures and traditional ML approaches.

## 📋 Project Overview

This project implements and compares **6 different models** for breast cancer classification using the **BreaKHis dataset** (Breast Cancer Histopathological Database). The goal is to classify histopathological images as either **benign** or **malignant** with high accuracy and reliability.

### 🎯 Key Features
- 6 different model architectures (CNN, Transfer Learning, Ensemble)
- High-quality visualizations (300 DPI) ready for publications
- Complete training and evaluation pipeline
- Reproducible results with detailed documentation
- Easy-to-use scripts for all tasks

## 🏆 Models Implemented

### Model 1: Transfer Learning + SVM
- **Model 1a**: AlexNet feature extraction + SVM classifier
  - Accuracy: 89.37% | AUC: 0.9687
- **Model 1b**: VGG16 feature extraction + SVM classifier
  - Accuracy: 90.87% | AUC: 0.9679

### Model 2: ResNet50 Transfer Learning
- Fine-tuned ResNet50 with custom classifier
- End-to-end deep learning approach

### Model 3: Ensemble CNN
- Custom ensemble architecture combining multiple CNN backbones
- Leverages diversity for improved predictions

### Model 4: Custom CNN Architecture
- Built-from-scratch CNN optimized for histopathological images
- Lightweight and efficient design

### Model 5: Inception + Xception Ensemble
- Combines Inception and Xception architectures
- State-of-the-art feature extraction

### Model 6: Handcrafted Features + ANN
- Traditional feature extraction (texture, shape, color)
- Artificial Neural Network classifier

## 📂 Project Structure

```
breast-cancer-classification/
├── data/
│   ├── processed/           # Train/val/test CSV splits
│   │   ├── train_100X.csv
│   │   ├── validation_100X.csv
│   │   └── test_100X.csv
│   ├── raw/                 # BreaKHis images (not included)
│   └── splits_info.json     # Split metadata
│
├── models/                  # Saved model artifacts
│   ├── paper1_alexnet_svm/
│   │   ├── metrics.json
│   │   ├── feature_svm_pipeline.joblib
│   │   └── README.md
│   ├── paper1_vgg16_svm/
│   ├── model2_resnet50_transfer/
│   └── ...
│
├── results/                 # Evaluation results
│   ├── model 1/             # Model 1a & 1b results
│   │   ├── model_1a_final_confusion_matrix.png
│   │   ├── model_1a_final_roc_curve.png
│   │   ├── model_1b_final_confusion_matrix.png
│   │   ├── model_1b_final_roc_curve.png
│   │   ├── model1_comparison.png
│   │   └── README.md
│   ├── model 2/
│   ├── model 3/
│   └── metrics_summary_from_preds.json
│
├── scripts/                 # Training and evaluation scripts
│   ├── model1_alexnet_vgg_svm.py
│   ├── model2_pytorch.py
│   ├── model3_pytorch.py
│   ├── generate_model1_visualizations.py
│   └── ...
│
├── src/                     # Source code modules
│   ├── data/               # Data loading utilities
│   ├── evaluation/         # Metrics and evaluation
│   ├── training/           # Training loops
│   └── utils/              # Helper functions
│       └── path_resolver.py
│
├── environment.yml         # Conda environment
├── requirements.txt        # Python dependencies
├── pyrightconfig.json     # Python type checking config
├── LICENSE                # MIT License
└── README.md              # This file
```

## 🚀 Quick Start Guide

### Prerequisites

Before you begin, ensure you have:
- **Python 3.11+** installed
- **CUDA-capable GPU** (recommended for training, but CPU works too)
- **Git** installed on your system
- At least **10 GB** of free disk space (for dataset and models)

### Step-by-Step Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Nagarohit29/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification
```

#### Step 2: Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate breast_cancer
```

**Option B: Using pip + venv**
```bash
# Create virtual environment
python -m venv breast_cancer_env

# Activate the environment
# On Windows:
breast_cancer_env\Scripts\activate
# On Linux/Mac:
source breast_cancer_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Download the BreaKHis Dataset

1. Visit the [BreaKHis Official Website](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
2. Download the dataset (approximately 2 GB)
3. Extract the downloaded file
4. Create the data directory structure:
   ```bash
   # Windows PowerShell
   New-Item -ItemType Directory -Force -Path "data\raw"
   
   # Linux/Mac
   mkdir -p data/raw
   ```
5. Move the extracted `BreaKHis_v1` folder to `data/raw/`
   
   Your structure should look like:
   ```
   data/raw/BreaKHis_v1/
   ├── histology_slides/
   │   └── breast/
   │       ├── benign/
   │       └── malignant/
   ```

#### Step 4: Verify Installation

Check that everything is installed correctly:

```bash
# Verify Python packages
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

# Check if GPU is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 📊 Dataset Preparation

The data splits (train/validation/test) are **already provided** in `data/processed/`:
- `train_100X.csv` - Training set
- `validation_100X.csv` - Validation set  
- `test_100X.csv` - Test set

**If you need to regenerate splits:**
```bash
python -m src.data_splitter
```

## 🎯 How to Run Models

### Training Models

#### Model 1: Transfer Learning + SVM (AlexNet & VGG16)

This model uses pretrained CNN features + SVM classifier. Training is relatively fast.

```bash
# Train both Model 1a (AlexNet) and Model 1b (VGG16)
python scripts/model1_alexnet_vgg_svm.py

# Expected runtime: ~20-30 minutes on GPU, ~1-2 hours on CPU
```

**What it does:**
1. Loads pretrained AlexNet and VGG16 models
2. Extracts features from images
3. Applies PCA for dimensionality reduction
4. Trains SVM classifier with RBF kernel
5. Saves models to `models/paper1_alexnet_svm/` and `models/paper1_vgg16_svm/`
6. Generates metrics and ROC curves

#### Model 2: ResNet50 Transfer Learning

Fine-tuned ResNet50 for end-to-end deep learning.

```bash
# Basic training
python scripts/model2_pytorch.py

# With custom parameters
python scripts/model2_pytorch.py --epochs 50 --batch-size 32 --lr 0.001

# Expected runtime: ~2-3 hours on GPU
```

**Available arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use ('cuda' or 'cpu')

#### Model 3: Ensemble CNN

Custom ensemble architecture combining multiple backbones.

```bash
python scripts/model3_pytorch.py

# Expected runtime: ~3-4 hours on GPU
```

#### Model 4: Custom CNN Architecture

Lightweight CNN built from scratch.

```bash
python scripts/model4_pytorch_clean.py

# Expected runtime: ~1-2 hours on GPU
```

#### Model 6: Handcrafted Features + ANN

Traditional feature extraction with neural network.

```bash
python scripts/model6_pytorch_fixed.py

# Expected runtime: ~30-45 minutes
```

### Generating Visualizations

After training, generate publication-quality visualizations (300 DPI).

```bash
# Model 1 visualizations (confusion matrices + ROC curves)
python scripts/generate_model1_visualizations.py

# Model 2 visualizations
python scripts/generate_final_visualizations.py

# Model 3 & 6 visualizations
python scripts/generate_model3_6_visualizations.py
```

Visualizations will be saved to respective `results/` subdirectories.

### Computing Metrics

If you have prediction CSV files and want to recompute metrics:

```bash
# Compute metrics from predictions
python scripts/compute_metrics_from_predictions.py

# Generate summary across all models
python scripts/generate_metrics_summary.py
```

### Running Inference on New Images

To classify new images using a trained model:

```python
# Example for Model 1a (AlexNet + SVM)
from joblib import load
import torch
from PIL import Image
from torchvision import transforms

# Load model
pipeline = load('models/paper1_alexnet_svm/feature_svm_pipeline.joblib')

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/your/image.png')
image_tensor = transform(image).unsqueeze(0)

# Extract features and predict
# (Feature extraction code needed - see scripts/model1_alexnet_vgg_svm.py)
prediction = pipeline.predict(features)
print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
```

## 📊 Results Summary

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| Model 1a (AlexNet+SVM) | 89.37% | 0.9687 | 97.26% | 87.27% | 0.9200 |
| Model 1b (VGG16+SVM) | **90.87%** | 0.9679 | 96.86% | **89.88%** | **0.9324** |
| Model 2 (ResNet50) | - | - | - | - | - |
| Model 3 (Ensemble) | - | - | - | - | - |

*Full results available in `results/` directory*

## 🔬 Methodology

1. **Data Preprocessing**
   - Images resized to 224×224
   - Normalization using ImageNet statistics
   - Data augmentation (random flip, rotation, color jitter)

2. **Feature Extraction (Model 1)**
   - Pretrained CNN backbones (AlexNet/VGG16)
   - Features extracted from final convolutional layer
   - PCA dimensionality reduction (512 components)

3. **Classification**
   - SVM with RBF kernel (Model 1)
   - Deep neural networks (Models 2-6)
   - Ensemble averaging (Model 3, 5)

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - AUC-ROC curve
   - Confusion matrices
   - Per-class performance

## 📈 Visualization Examples

All visualizations are high-resolution (300 DPI, 10×8 inches) for publication quality.

- Confusion matrices with percentage annotations
- ROC curves with AUC scores
- Model comparison charts
- Performance metrics summaries

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Machine Learning**: scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Computer Vision**: PIL, OpenCV
- **Utilities**: tqdm, joblib

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{breast-cancer-classification-2025,
  author = {Nagarohit},
  title = {Breast Cancer Classification Using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Nagarohit29/Breast-Cancer-Classification}
}
```

**BreaKHis Dataset Citation:**
```bibtex
@article{spanhol2016breast,
  title={A dataset for breast cancer histopathological image classification},
  author={Spanhol, Fabio A and Oliveira, Luiz S and Petitjean, Caroline and Heutte, Laurent},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={63},
  number={7},
  pages={1455--1462},
  year={2016},
  publisher={IEEE}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

### Dataset Not Included
The BreaKHis dataset is **NOT included** in this repository due to its large size (~2 GB). You must download it separately from the [official source](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).

### Model Checkpoints
Large PyTorch model checkpoints (`.pth` files) are excluded from Git. After training, these will be saved locally in `models/` directory. The SVM models (`.joblib` files) are included as they are smaller.

### GPU vs CPU Training
- **With GPU**: Most models train in 1-4 hours
- **Without GPU**: Training may take 5-10x longer (5-20 hours per model)
- Inference and feature extraction work well on CPU

### Memory Requirements
- Minimum 8 GB RAM recommended
- For batch training: 16 GB RAM + 6 GB GPU memory ideal

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
python scripts/model2_pytorch.py --batch-size 16  # instead of 32
```

### Issue: Dataset path not found
**Solution**: Verify BreaKHis is in correct location
```bash
# Should exist:
data/raw/BreaKHis_v1/histology_slides/breast/
```

### Issue: Module not found errors
**Solution**: Install missing packages
```bash
pip install -r requirements.txt
```

### Issue: Low accuracy during training
**Solution**: 
- Ensure dataset is properly extracted
- Check that data splits are loaded correctly
- Verify image preprocessing matches expected format
- Try adjusting learning rate or epochs

## 📞 Support & Contact

- **Issues**: Open an issue on [GitHub Issues](https://github.com/Nagarohit29/Breast-Cancer-Classification/issues)
- **Discussions**: Start a discussion for questions
- **Email**: [Contact maintainer]

## 🙏 Acknowledgments

- **BreaKHis Dataset**: Federal University of Paraná (UFPR), Brazil
- **PyTorch Team**: For the excellent deep learning framework
- **scikit-learn**: For machine learning utilities
- **Open Source Community**: For various tools and libraries used

## 📚 References & Related Work

1. **BreaKHis Dataset Paper**:
   - Spanhol et al. (2016) - "A dataset for breast cancer histopathological image classification"
   - IEEE Transactions on Biomedical Engineering

2. **Transfer Learning**:
   - Pre-trained models from ImageNet classification

3. **Ensemble Methods**:
   - Combining multiple model predictions for improved accuracy

## 🚀 Future Work

- [ ] Add Model 5 training script
- [ ] Implement k-fold cross-validation
- [ ] Add Grad-CAM visualizations for explainability
- [ ] Support for other magnification factors (40X, 200X, 400X)
- [ ] Web interface for easy inference
- [ ] Docker containerization
- [ ] Model compression and optimization
- [ ] Integration with medical imaging standards (DICOM)

## ⚠️ Medical Disclaimer

**IMPORTANT**: This project is for **research and educational purposes only**. 

- It should **NOT** be used for clinical diagnosis without proper validation
- Medical decisions should always be made by qualified healthcare professionals
- No warranty or guarantee of accuracy is provided
- This is an experimental research tool, not a certified medical device

## 📊 Performance Benchmarks

Training times measured on NVIDIA RTX 3080 (10GB):

| Model | Training Time | Inference (per image) | Model Size |
|-------|--------------|----------------------|------------|
| Model 1a (AlexNet+SVM) | ~20 min | ~50 ms | ~15 MB |
| Model 1b (VGG16+SVM) | ~25 min | ~70 ms | ~40 MB |
| Model 2 (ResNet50) | ~2.5 hours | ~30 ms | ~95 MB |
| Model 3 (Ensemble) | ~3.5 hours | ~100 ms | ~180 MB |
| Model 4 (Custom CNN) | ~1.5 hours | ~20 ms | ~25 MB |
| Model 6 (Handcrafted) | ~40 min | ~15 ms | ~5 MB |

## 🌟 Star This Repository

If you find this project helpful, please consider giving it a ⭐ on GitHub!

---

**Project Status**: ✅ Active Development  
**Last Updated**: November 7, 2025  
**Version**: 1.0.0

---

Made with ❤️ for advancing breast cancer detection research
