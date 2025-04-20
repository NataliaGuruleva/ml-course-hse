# ml-course-hse
Machine Learning course materials

---

# Homework 1: k-NN, KD-tree

This homework contains an implementation of the **KNN** algorithm optimized with a **KD-Tree** and testing on two datasets:
1. **Cancer Dataset**: Contains tumor features labeled as `M` (benign) or `B` (malignant).
2. **Spam Dataset**: Contains email features labeled as `1` (spam) or `0` (not spam).

The implementation includes building a KD-Tree for efficient nearest-neighbor searches and evaluating model performance using precision, recall, and accuracy metrics.

## Tasks

  - `Dataset Loading`
  - `Train-Test Split`
  - `Metrics: precision, recall, and accuracy for each class`
  - `KD-Tree Implementation`
  - `K-NN Classifier`

---

# Homework 2: Clusterization

This homework contains implementations of three clustering algorithms:
1. **K-Means** - Centroid-based clustering with different initialization methods
2. **DBSCAN** - Density-based spatial clustering
3. **Agglomerative Clustering** - Hierarchical clustering

---

# Homework 3: Linear Regression

This homework contains implementations of linear regression algorithms with feature importance analysis.

## Tasks

  - `Evaluation Metrics: MSE, R2`
  - `Linear Regression Models: using analytical solution and gradient descent with L1 regularization`
  - `Feature Importance`

---

# Homework 4: Perceptrons

This homework contains two implementations of perceptron algorithms and image transformation method using perceptron model.

## Tasks

  - `Basic Perceptron`
  - `Improved Perceptron (tracks weights with best accuracy during training)`
  - `Image Processing (transformation that converts images to 2D feature vectors)`

---

# Homework 5: Neural Networks

This homework contains implementations of neural network components and models using NumPy only; using PyTorch.

## Tasks

  - `Neural Network Modules (NumPy): Linear, ReLU`
  - `Multi-Layer Perceptron (MLP) Classifier - architecture through module list`
  - `PyTorch CNN Implementation`

---

# Homework 6: Variational Autoencoder (VAE)

This homework contains an implementation of a Variational Autoencoder using PyTorch for image generation and learning on cats dataset.

## Tasks

  - `Encoder Network`
  - `Decoder Network`
  - `VAE Model`

---

# Homework 7: Support Vector Machines (SVM)

This homework contains implementations of Support Vector Machines with linear and kernel methods for binary classification.

## Tasks

  - `Linear SVM`
  - `Kernel SVM: polynomial kernel, gaussian (RBF) kernel`

---

# Homework 8: Decision Tree Classifier

This homework contains an implementation of a Decision Tree Classifier with both Gini impurity and entropy criteria.

## Tasks

  - `Impurity Measures: gini, entropy`
  - `Information gain for splits`
  - `DecisionTreeLeaf: contains class distribution`
  - `DecisionTreeNode: contains split dimension`
  - `Decision Tree Classifier`

---

# Homework 9: Random Forest and CatBoost Classifiers

This homework contains implementations of ensemble methods including Random Forest and pre-trained CatBoost models for classification tasks.

## Tasks
  - `Decision Tree Model`
  - `Random Forest Classifier`
  - `Random ForestClassifier with fFeature importance calculation`
  - `Pre-trained CatBoost gradient boosting models: Age prediction model (200 trees, max_depth=10), Gender prediction model (200 trees, max_depth=7)`

---
