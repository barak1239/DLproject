# Skin Lesion Classification: A Comparative Study of Multiple ML Models

## 1. Introduction
This project explores the HAM10000 dataset of dermatoscopic images, applying multiple machine learning models to classify various skin lesions. We compare four approaches—Baseline (majority-class), Logistic Regression, a Basic Neural Network, and a CNN—to test the hypothesis that increasing model complexity leads to better performance and more localized, clinically relevant feature extraction.

## 2. Dataset Overview
- **Source:** The HAM10000 (“Human Against Machine with 10000 training images”) dataset.
- **Classes:** Each image is labeled with one of seven lesion types (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vas`).
- **Metadata:** Contains additional patient info (age, sex, localization) for further analysis.

### 2.1 Data Visualization
#### Class Distribution
**Figure 1 (Diagnosis Distribution):**
- Shows a bar chart of numeric labels (0–6) or `dx` codes. 0-6 is (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vas`). with the same order
- Clear imbalance: label 0.0 (often “nv”) is the majority class (~7000 samples), while other classes have far fewer images.

#### Representative Images
**Figures 2–9 (Sample Images per Class):**
- Each figure shows three images side by side, labeled by `image_id`.
- Suptitle indicates the lesion class.
- Example: In Figure 2, we see `ISIC_0024646`, `ISIC_0027753`, and `ISIC_0032404` from the same class.
- Helps visually confirm the variety of lesion appearances (color, texture, borders).

#### Grad-CAM Example
**Figure 10 (Grad-CAM on 28×28 normalized image):**
- Displays original image, Grad-CAM heatmap, and overlay.
- The CNN model focuses on the lower region in the heatmap, indicating which pixels contributed most to its prediction.

## 3. Hypothesis
Different classes of machine learning models—from simple linear classifiers to deep convolutional neural networks—capture distinct aspects of the underlying data distribution in skin lesion images.
We hypothesize that simpler models (like logistic regression) primarily learn global linear separability, whereas more complex architectures (basic NNs, CNNs) uncover hierarchical, non-linear representations aligned with subtle pathological features.


### Key Points
- **Performance Gains:** Deep models capture fine-grained, localized features, crucial for differentiating benign vs. malignant lesions.
- **Model Progression:** Baseline → Logistic Regression → Basic NN → CNN reveals a shift from coarse global patterns to nuanced, spatially-distributed features.
- **Interpretability:** Grad-CAM visualizes CNN focus, showing advanced models attend to clinically relevant regions (lesion boundaries, color changes), whereas simpler models do not.

## 4. Models and Methods
### Baseline (Majority Class)
- Predicts the most frequent lesion type in all cases.
- Serves as a minimal benchmark.

### Logistic Regression (LG)
- A linear model that attempts to separate classes in a high-dimensional space of raw or preprocessed pixel features.
- Captures global, linear patterns but struggles with non-linearities.

### Basic Neural Network (NN)
- A feedforward architecture with hidden layers, introducing non-linear activations to learn more complex relationships.
- Better than LG at handling moderate non-linearities but still lacks localized feature extraction like CNNs.

### Convolutional Neural Network (CNN)
- Designed for image tasks, uses convolution and pooling to learn hierarchical features.
- We apply data augmentation, batch normalization, and interpretability via Grad-CAM.

### 4.1 Training Setup
- **Splits:** Data is divided into training, validation, and test sets.
- **Metrics:** Accuracy, precision, recall, F1-score, confusion matrices.
- **Visualization:** Training curves (loss, accuracy) and validation performance to ensure neither underfitting nor overfitting.

## 5. Results
Below are screenshots of model training and performance logs (via Weights & Biases or similar tracking tools). The x-axis represents training steps or epochs; the y-axis represents accuracy, loss, precision, recall, etc.

### Baseline
**Figure 11:**
- Shows minimal improvement in metrics (e.g., accuracy ~ majority class proportion).
- Confirms strong class imbalance (dominant “nv” class).

### Logistic Regression (LG)
**Figure 12:**
- Logs indicate moderate accuracy gains, improved over baseline but still limited.
- Can handle some linear separability but struggles with complex textures or color boundaries.

### Basic Neural Network (NN)
**Figure 13:**
- Training curves (train_accuracy, val_accuracy, train_loss, val_loss) show steady improvement.
- Final accuracy higher than LG, reflecting NN’s ability to learn non-linear feature combinations.

### CNN
**Figure 14:**
- Best overall performance.
- `val_accuracy` climbs steadily, `train_loss` decreases significantly.
- Achieves the highest test accuracy, effectively learning localized patterns.
- Grad-CAM (Figure 10) confirms CNN's focus on relevant lesion regions.

### 5.1 Observations
- **Performance vs. Complexity:** CNN outperforms simpler models. Baseline sets a low bar, LG improves, NN does better, and CNN performs best.
- **Loss & Accuracy Trends:** CNN’s loss drops sharply, accuracy climbs higher than other models, showing effective feature extraction.
- **Interpretability:** Grad-CAM confirms CNN attends to clinically meaningful regions. Simpler models lack clear localization.

## 6. Discussion
### Model Complexity Spectrum
- **Baseline & LG:** Rely on global, often linear signals.
- **NN:** Introduces non-linearity but lacks spatial “awareness.”
- **CNN:** Captures hierarchical, localized features, critical for lesion diagnosis.

### Latent Representation & Clinical Relevance
- Grad-CAM overlays show CNN “highlighting” lesion boundaries and color variations, aligning with dermatological diagnostic markers.

### Comparative Insight
- Confusion matrices indicate advanced architectures reduce misclassification.
- Increasing model complexity improves classification but requires more computational resources.

### Theoretical Implications
- Representation learning in CNNs is better suited for complex image tasks like skin lesion analysis.
- The shift from global to local feature extraction generalizes to other medical imaging domains.

## 7. Conclusion
This project demonstrates how model complexity affects skin lesion classification:

| Model | Key Takeaways |
|--------|----------------|
| **Baseline** | Exposes class imbalance, naive predictor. |
| **Logistic Regression** | Improves upon baseline but limited by linear assumptions. |
| **Basic NN** | Learns non-linear relationships, boosting performance. |
| **CNN** | Delivers best results, leveraging localized feature extraction. Grad-CAM confirms focus on clinically relevant regions. |

### Takeaways
- Deep learning significantly enhances classification accuracy for skin lesions.
- Interpretability (e.g., Grad-CAM) is crucial in clinical contexts.
- **Future Work:** More advanced architectures (e.g., Transformers, ensembles), integration of clinical data, or refined interpretability tools.

## 8. References & Further Reading
- **HAM10000 Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Grad-CAM:** Selvaraju, R. R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.” ICCV, 2017.
- **Representation Learning:** Bengio, Y., et al. “Representation Learning: A Review and New Perspectives.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013.

## Appendix: Figures
| Figure | Description |
|--------|-------------|
| **Figure 1** | Diagnosis distribution (bar chart). |
| **Figures 2-9** | Sample images for each class. |
| **Figure 10** | Grad-CAM visualization. |
| **Figure 11** | Baseline Model Logs. |
| **Figure 12** | Logistic Regression Logs. |
| **Figure 13** | Basic Neural Network Logs. |
| **Figure 14** | CNN Logs. |

Each set of logs confirms the correlation between model complexity and classification accuracy.




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



