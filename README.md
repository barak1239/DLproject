# Skin Lesion Classification: A Comparative Study of Multiple ML Models

## 1. Introduction
This project explores the HAM10000 dataset of dermatoscopic images, applying multiple machine learning models to classify various skin lesions. We compare four approaches:
- **Baseline (majority-class)**
- **Logistic Regression**
- **Basic Neural Network**
- **Convolutional Neural Network (CNN)**

We test the hypothesis that increasing model complexity leads to better performance and more localized, clinically relevant feature extraction.

## 2. Dataset Overview
**Source:** The HAM10000 dataset.

**Classes:** Each image is labeled with one of seven lesion types:
- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vas)

**Metadata:** The dataset contains additional patient info (age, sex, localization) for further analysis.

### 2.1 Data Visualization
**Class Distribution:**
- **Figure 1:** A bar chart showing numeric labels (0–6) or diagnosis codes, revealing a strong imbalance. Label 0 ("nv") is the majority class (~7000 samples), while others are underrepresented.

**Representative Images:**
- **Figures 2–9:** Sample images from each lesion class, displayed three at a time.

**Grad-CAM Example:**
- **Figure 10:** Demonstrates how the CNN highlights crucial lesion regions in images.

## 3. Hypothesis
Different machine learning models capture distinct aspects of skin lesion images:
- **Logistic Regression** learns global linear separability.
- **Basic NN** captures non-linear relationships but lacks spatial awareness.
- **CNN** extracts hierarchical, localized features, aligning with pathological patterns.

### Key Insights:
- **Performance Gains:** Deep models capture fine-grained details.
- **Model Progression:** Increasing complexity shifts feature extraction from global to nuanced, spatially-distributed patterns.
- **Interpretability:** Grad-CAM highlights crucial lesion boundaries, unlike simpler models.

## 4. Models and Methods
### Baseline (Majority Class)
- Predicts the most frequent lesion type in all cases.
- Serves as a minimal benchmark.

### Logistic Regression (LG)
- A linear model that separates classes in high-dimensional space.
- Captures global patterns but struggles with non-linearities.

### Basic Neural Network (NN)
- A feedforward network with hidden layers and non-linear activations.
- Handles moderate non-linearities but lacks spatial feature extraction.

### Convolutional Neural Network (CNN)
- Uses convolution and pooling to learn hierarchical features.
- Includes data augmentation, batch normalization, and Grad-CAM interpretability.

### 4.1 Training Setup
- **Splits:** Training, validation, and test sets.
- **Metrics:** Accuracy, precision, recall, F1-score, confusion matrices.
- **Visualization:** Training curves to monitor underfitting/overfitting.

## 5. Results
Training logs tracked using Weights & Biases (WandB) or similar platforms.

### **Baseline**
- **Figure 11:** Minimal improvement in metrics, confirming strong class imbalance.

### **Logistic Regression**
- **Figure 12:** Moderate accuracy gains, limited by linear assumptions.

### **Basic Neural Network**
- **Figure 13:** Training curves show steady improvement; outperforms LG.

### **CNN**
- **Figure 14:** Best performance. The validation accuracy improves significantly.
- **Grad-CAM (Figure 10):** Highlights lesion regions, confirming interpretability.

### **5.1 Observations**
- **Performance vs. Complexity:** CNN outperforms simpler models.
- **Loss & Accuracy Trends:** CNN loss decreases sharply; accuracy improves steadily.
- **Interpretability:** Grad-CAM visualizations confirm CNN attends to meaningful regions.

## 6. Discussion
### Model Complexity Spectrum
- **Baseline & LG:** Rely on global, often linear signals.
- **NN:** Introduces non-linearity but lacks spatial awareness.
- **CNN:** Extracts localized features, crucial for lesion diagnosis.

### Latent Representation & Clinical Relevance
- **Grad-CAM overlays** align with dermatological diagnostic markers.

### Comparative Insight
- **Confusion matrices** show CNN reduces misclassification.
- **Increased complexity yields better classification** but demands higher computational resources.

### Theoretical Implications
- CNN representation learning generalizes to other medical imaging tasks.

## 7. Conclusion
This study demonstrates the relationship between model complexity and skin lesion classification accuracy:
- **Baseline:** Highlights class imbalance, naive predictor.
- **Logistic Regression:** Limited by linear separability.
- **Basic NN:** Captures non-linear features, improving performance.
- **CNN:** Achieves best accuracy, leveraging localized feature extraction.

### Key Takeaways
- Deep learning significantly improves skin lesion classification.
- Grad-CAM enhances interpretability in clinical contexts.
- Future work may explore advanced architectures (Transformers, ensembles) and better integration of clinical metadata.

## 8. References & Further Reading
- **HAM10000 Dataset:** [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Grad-CAM:** Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV, 2017.*
- **Representation Learning:** Bengio, Y., et al. "Representation learning: A review and new perspectives." *IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013.*

## Appendix: Figures
| Figure | Description |
|--------|-------------|
| **Figure 1** | Diagnosis distribution (bar chart). |
| **Figures 2-9** | Sample images for each class. |
| **Figure 10** | Grad-CAM visualization. |
| **Figure 11** | Baseline Model Logs (accuracy, precision, recall, etc.). |
| **Figure 12** | Logistic Regression Logs. |
| **Figure 13** | Basic Neural Network Logs. |
| **Figure 14** | CNN Logs (train_loss, val_loss, train_accuracy, val_accuracy, best_val_accuracy, test_accuracy). |

Each set of logs illustrates model performance trends, confirming the correlation between model complexity and classification accuracy.



![image](https://github.com/user-attachments/assets/d9932b3b-e354-43dd-8d95-44ed25651257)
![image](https://github.com/user-attachments/assets/e214e3e1-7fa4-4be4-8ae0-c561a94fe646)
![image](https://github.com/user-attachments/assets/db65f277-4603-401d-82e8-e5d565f70e77)
![image](https://github.com/user-attachments/assets/0184e65d-085e-44e7-bef0-bfb0c1a5d1b5)
![image](https://github.com/user-attachments/assets/c98f0e1b-adb4-43ab-a002-93ae5d9cd467)
![image](https://github.com/user-attachments/assets/4f4fd88c-e0fa-4455-a446-e64f28cfcc14)
![image](https://github.com/user-attachments/assets/450ccaba-b8df-4b48-8f8e-f8cde6a2b81e)
![image](https://github.com/user-attachments/assets/db6e982b-65b1-4bbe-9435-8b2441ba09bc)
![image](https://github.com/user-attachments/assets/c07a19be-973a-4261-addd-b60168321d2e)
![image](https://github.com/user-attachments/assets/8d6b07f4-6c1d-41b4-86ec-6b1c45ec8198)
![image](https://github.com/user-attachments/assets/e922dfa7-66d9-40ba-9ac7-66f69d5154c0)
![image](https://github.com/user-attachments/assets/2f75617a-dd4e-47ff-b0d3-3ed18a5ec0b8)
![image](https://github.com/user-attachments/assets/c13d3043-d2c8-4855-b2e7-c6949e24e697)



