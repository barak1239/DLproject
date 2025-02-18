![image](https://github.com/user-attachments/assets/b9ab08d2-887b-4678-a802-d1b5f80c4514)Skin Lesion Classification: A Comparative Study of Multiple ML Models
1. Introduction
This project explores the HAM10000 dataset of dermatoscopic images, applying multiple machine learning models to classify various skin lesions. We compare four approaches—Baseline (majority-class), Logistic Regression, a Basic Neural Network, and a CNN—to test the hypothesis that increasing model complexity leads to better performance and more localized, clinically relevant feature extraction.

2. Dataset Overview
Source: The HAM10000 (“Human Against Machine with 10000 training images”) dataset.
Classes: Each image is labeled with one of seven lesion types (akiec, bcc, bkl, df, mel, nv, vas).
Metadata: Contains additional patient info (age, sex, localization) for further analysis.
2.1 Data Visualization
Class Distribution

Figure 1 (Diagnosis Distribution): Shows a bar chart of numeric labels (0–6) or dx codes. You can see a clear imbalance: label 0.0 (often “nv”) is the majority class with around 7000 samples, while other classes have far fewer images.
Representative Images

Figures 2–9 (Sample Images per Class): Each figure shows three images side by side, labeled by image_id. Above these images is a suptitle indicating the lesion class.
For instance, in Figure 2, we see IDs ISIC_0024646, ISIC_0027753, and ISIC_0032404 from the same class.
Each subsequent figure (3–9) similarly shows sample images from other classes. This helps us visually confirm the variety of lesion appearances (color, texture, borders).
Grad-CAM Example

Figure 10 (Grad-CAM on 28×28 normalized image): Illustrates the original image, the Grad-CAM heatmap, and the overlay. The model (CNN) focuses on the lower region in the heatmap, indicating which pixels contributed most to its prediction.
3. Hypothesis
Different classes of machine learning models—from simple linear classifiers to deep convolutional neural networks—capture distinct aspects of the underlying data distribution in skin lesion images.
We hypothesize that simpler models (like logistic regression) primarily learn global linear separability, whereas more complex architectures (basic NNs, CNNs) uncover hierarchical, non-linear representations aligned with subtle pathological features.

Key Points
Performance Gains: Deep models capture fine-grained, localized features, crucial for differentiating benign vs. malignant lesions.
Model Progression: Baseline → Logistic Regression → Basic NN → CNN reveals a shift from coarse global patterns to more nuanced, spatially-distributed features.
Interpretability: Grad-CAM helps visualize the CNN’s focus, showing that advanced models attend to clinically relevant regions (lesion boundaries, color changes), whereas simpler models do not explicitly localize features.
4. Models and Methods
Baseline (Majority Class)

Predicts the most frequent lesion type in all cases.
Serves as a minimal benchmark.
Logistic Regression (LG)

A linear model that attempts to separate classes in a high-dimensional space of raw or preprocessed pixel features.
Captures global, linear patterns but struggles with non-linearities.
Basic Neural Network (NN)

A feedforward architecture with hidden layers, introducing non-linear activations to learn more complex relationships.
Better than LG at handling moderate non-linearities but still lacks localized feature extraction like CNNs.
Convolutional Neural Network (CNN)

Designed for image tasks, uses convolution and pooling to learn hierarchical features.
We apply data augmentation, batch normalization, and interpretability via Grad-CAM.
4.1 Training Setup
Splits: Data is typically divided into training, validation, and test sets.
Metrics: Accuracy, precision, recall, F1-score, confusion matrices.
Visualization: We track training curves (loss, accuracy) and validation performance to ensure we are neither underfitting nor overfitting.
5. Results
Below are screenshots of the model training and performance logs for each approach, as tracked via Weights & Biases (WandB) or a similar platform. The x-axis typically represents training steps or epochs; the y-axis represents metrics like accuracy, loss, precision, recall, etc.

Baseline

Figure 11: Shows minimal improvement in metrics (e.g., accuracy ~ the proportion of the majority class).
Achieves the lowest performance but confirms the presence of strong class imbalance (dominant “nv” class).
Logistic Regression (LG)

Figure 12: The logs indicate moderate accuracy gains, improved over baseline but still limited.
The model can handle some linear separability in the data but cannot capture complex textures or color boundaries.
Basic Neural Network (NN)

Figure 13: Training curves (train_accuracy, val_accuracy, train_loss, val_loss) show steady improvement.
Final accuracy is higher than LG, reflecting the NN’s ability to learn non-linear feature combinations.
CNN

Figure 14: We see the best overall performance. The val_accuracy climbs steadily, and train_loss decreases significantly.
Achieves the highest test accuracy, indicating effective learning of localized patterns.
Grad-CAM (Figure 10) reveals the CNN focusing on relevant lesion regions, highlighting the interpretability advantage of advanced models.
5.1 Observations
Performance vs. Complexity: As hypothesized, the CNN outperforms simpler models. The baseline sets a low bar, LG improves upon it, the NN does better still, and the CNN performs best.
Loss & Accuracy Trends: The CNN’s loss drops more sharply, and its accuracy climbs higher than other models, showing it effectively extracts crucial image features.
Interpretability: Grad-CAM visualizations confirm the CNN attends to clinically meaningful regions. Simpler models do not offer such clear localization.
6. Discussion
Model Complexity Spectrum

Baseline and LG rely on global, often linear signals.
The NN introduces non-linearity but still lacks spatial “awareness.”
The CNN captures hierarchical, localized features, critical for lesion diagnosis.
Latent Representation & Clinical Relevance

Grad-CAM overlays show the CNN “highlighting” lesion boundaries and color variations, aligning with dermatological diagnostic markers.
Comparative Insight

The difference in confusion matrices across models indicates that advanced architectures reduce misclassification.
Each increment in complexity yields better classification, though at the cost of more computational resources.
Theoretical Implications

Our findings support the notion that representation learning in CNNs is better suited for complex image tasks (like skin lesion analysis).
The shift from global to local feature extraction can generalize to other medical imaging domains.
7. Conclusion
This project demonstrates how model complexity affects skin lesion classification:

Baseline: Exposes the strong class imbalance, acts as a naive predictor.
Logistic Regression: Improves upon baseline but limited by linear assumptions.
Basic NN: Learns non-linear relationships, further boosting performance.
CNN: Delivers the best results, leveraging localized feature extraction. Grad-CAM confirms the model’s focus on clinically relevant lesion regions.
Takeaways:

Deep learning significantly enhances classification accuracy for skin lesions.
Interpretability (e.g., Grad-CAM) is essential in clinical contexts to verify the model is looking at the right features.
Future Work may involve more advanced architectures (e.g., Transformers, ensembles), integration of clinical data, or refined interpretability tools.
8. References & Further Reading
HAM10000 Dataset: Kaggle Link
Grad-CAM: Selvaraju, R. R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.” ICCV, 2017.
Representation Learning: Bengio, Y., et al. “Representation learning: A review and new perspectives.” IEEE transactions on pattern analysis and machine intelligence, 2013.
Appendix: Figures
Below is a quick reference for the figures described above:

Figure 1: Diagnosis Distribution (bar chart).
Figures 2–9: Representative images for each class, 3 images side by side with a single suptitle naming the class.
Figure 10: Baseline Model Logs (accuracy, precision, recall, etc.).
Figure 11: Logistic Regression Logs.
Figure 12: Basic Neural Network Logs.
Figure 13: CNN Logs (train_loss, val_loss, train_accuracy, val_accuracy, best_val_accuracy, test_accuracy).
Figure 14: Grad-CAM example, showing Original Image, Heatmap, and Overlay.
Each set of logs demonstrates the performance trends for that particular model, confirming our hypothesis about the correlation between model complexity and performance.

![image](https://github.com/user-attachments/assets/d9932b3b-e354-43dd-8d95-44ed25651257)
![image](https://github.com/user-attachments/assets/9e0efc26-6b6d-4b28-a15d-9bcd4d2f9b01)
![image](https://github.com/user-attachments/assets/f6e10536-4361-4fa8-8cab-2f8960b5a67d)
![image](https://github.com/user-attachments/assets/1ea1e41d-f92e-403f-b1dc-09dcb13e9427)
![image](https://github.com/user-attachments/assets/584ea47e-0a83-4891-8919-a0702b4590bc)
![image](https://github.com/user-attachments/assets/f0833c69-49b8-4d11-be19-7108c6a4ff47)
![image](https://github.com/user-attachments/assets/527dc514-2123-475e-b734-4923751f9153)
![image](https://github.com/user-attachments/assets/eb5b0545-c298-42b8-81bc-69d81c768258)
![image](https://github.com/user-attachments/assets/6c14be86-88d1-4d92-b07b-1cce01d7fa3d)
![image](https://github.com/user-attachments/assets/c07a19be-973a-4261-addd-b60168321d2e)
![image](https://github.com/user-attachments/assets/8d6b07f4-6c1d-41b4-86ec-6b1c45ec8198)
![image](https://github.com/user-attachments/assets/e922dfa7-66d9-40ba-9ac7-66f69d5154c0)
![image](https://github.com/user-attachments/assets/2f75617a-dd4e-47ff-b0d3-3ed18a5ec0b8)
![image](https://github.com/user-attachments/assets/c13d3043-d2c8-4855-b2e7-c6949e24e697)



