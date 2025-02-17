Different classes of machine learning models—from simple linear classifiers to deep convolutional neural networks—capture distinct aspects of the underlying data distribution in skin lesion images. We hypothesize that simpler models, such as logistic regression, primarily learn global linear separability based on aggregate image features, while more complex neural architectures (basic feedforward networks and CNNs) progressively uncover hierarchical, non-linear representations that align with intricate pathological features. By systematically comparing these approaches, we expect to demonstrate that:

• The performance improvements observed in deep models (especially CNNs) are a consequence of their ability to capture fine-grained, localized features that are critical for differentiating between benign and malignant lesions.

• The progression from baseline models to deep architectures reflects a transition from capturing coarse, global patterns to modeling subtle, spatially-distributed features, which are theoretically more representative of the complex biological processes underlying skin cancer.

• Interpretability methods (e.g., Grad-CAM for CNNs) can bridge the gap between model complexity and clinical relevance, revealing that the deep models focus on areas corresponding to known diagnostic markers, whereas simpler models may rely on less localized or intuitive features.
