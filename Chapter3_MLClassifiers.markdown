# Slide 1: Title Slide
**A Tour of Machine Learning Classifiers Using scikit-learn**  
*Chapter 3: Python Machine Learning (2nd Edition)*  
Sebastian Raschka & Vahid Mirjalili  
Prepared for: [Your Course/Workshop Name]  
Date: August 17, 2025

---

# Slide 2: Learning Objectives
- Understand key machine learning classifiers and their applications.
- Learn to implement classifiers using scikit-learn on the Iris dataset.
- Explore model evaluation, visualization, and hyperparameter tuning.
- Compare linear and nonlinear classifiers for different data types.

---

# Slide 3: Choosing a Classification Algorithm
- **No Free Lunch Theorem**: No single algorithm is best for all problems.
- Factors to consider:
  - Dataset size, noise, and linear separability.
  - Feature selection, performance metrics, and model tuning.
- Workflow:
  1. Select features.
  2. Choose performance metrics.
  3. Train and evaluate models.
  4. Tune hyperparameters.

---

# Slide 4: scikit-learn and the Perceptron
- **Perceptron**: Simple linear classifier, updates weights for misclassified samples.
- **scikit-learn Implementation**: `sklearn.linear_model.Perceptron`
- Example (Iris dataset, petal length/width):
  - Split data (70/30), standardize features, train model.
  - Accuracy: ~93% (linearly separable data).
- **Visualization**: Decision regions via `plot_decision_regions`.
- **Strengths**: Fast, simple for large datasets.
- **Weaknesses**: Converges only for linear data, no probabilities.

---

# Slide 5: Logistic Regression
- **Overview**: Linear model for binary/multiclass classification, uses sigmoid for probabilities.
- **Key Concepts**:
  - Logit function, odds ratio, logistic cost function.
  - L2 regularization (`C` = inverse of λ) to prevent overfitting.
- **scikit-learn**: `sklearn.linear_model.LogisticRegression` (OvR for multiclass).
- Example: Iris dataset, ~97% accuracy, `predict_proba` for probabilities.
- **Strengths**: Interpretable, robust to noise, probability outputs.
- **Weaknesses**: Assumes linear separability, sensitive to outliers.

---

# Slide 6: Support Vector Machines (SVM)
- **Linear SVM**: Maximizes margin using support vectors, handles errors via slack variables.
- **scikit-learn**: `sklearn.svm.SVC(kernel='linear')`.
- Example: Iris dataset, high accuracy with `C` controlling margin vs. error trade-off.
- **Strengths**: Effective in high dimensions, robust to overfitting.
- **Weaknesses**: Computationally intensive, requires feature scaling.

---

# Slide 7: Kernel SVM for Nonlinear Data
- **Kernel Trick**: Maps data to higher dimensions (e.g., RBF kernel).
- **Hyperparameters**:
  - `C`: Margin vs. error trade-off.
  - `gamma`: Controls nonlinearity in RBF kernel.
- Example: XOR, moons datasets; tunes `C` and `gamma`.
- **scikit-learn**: `sklearn.svm.SVC(kernel='rbf')`.
- **Strengths**: Handles complex patterns.
- **Weaknesses**: Overfitting risk with high `gamma`, less interpretable.

---

# Slide 8: Decision Tree Learning
- **Overview**: Splits data to maximize information gain (Gini/entropy).
- **scikit-learn**: `sklearn.tree.DecisionTreeClassifier`.
- Example: Iris dataset, visualizes tree and decision regions.
- **Hyperparameters**: `max_depth`, `min_samples_leaf` to prevent overfitting.
- **Strengths**: No scaling needed, interpretable, handles mixed data.
- **Weaknesses**: Overfits noisy data, unstable to small changes.

---

# Slide 9: Random Forests
- **Ensemble Method**: Combines multiple decision trees via majority vote.
- **scikit-learn**: `sklearn.ensemble.RandomForestClassifier`.
- Example: Iris dataset, ranks feature importance, uses OOB samples.
- **Hyperparameters**: `n_estimators`, `max_features`.
- **Strengths**: Robust, accurate, reduces overfitting.
- **Weaknesses**: Less interpretable, computationally heavier.

---

# Slide 10: K-Nearest Neighbors (KNN)
- **Overview**: Non-parametric, predicts via majority vote of k neighbors.
- **scikit-learn**: `sklearn.neighbors.KNeighborsClassifier`.
- Example: Iris dataset, tunes `n_neighbors` for bias-variance balance.
- **Hyperparameters**: `k`, distance metric (e.g., Euclidean).
- **Strengths**: Simple, adapts to local patterns.
- **Weaknesses**: Slow predictions, sensitive to scaling/irrelevant features.

---

# Slide 11: Practical Implementation with scikit-learn
- **Unified Workflow**:
  1. Load data (`datasets.load_iris`).
  2. Preprocess: Split (`train_test_split`), standardize (`StandardScaler`).
  3. Train (`fit`), predict (`predict`/`predict_proba`).
  4. Evaluate (`accuracy_score`), visualize (`plot_decision_regions`).
- **Hyperparameter Tuning**:
  - Logistic/SVM: `C`.
  - Kernel SVM: `gamma`.
  - KNN: `n_neighbors`.
- Consistent API across models simplifies experimentation.

---

# Slide 12: Key Takeaways
- **Linear Models**: Perceptron, Logistic Regression, Linear SVM.
  - Fast, interpretable, but limited to linear data.
- **Nonlinear Models**: Kernel SVM, Decision Trees, Random Forests, KNN.
  - Handle complex patterns, risk overfitting or high computation.
- **scikit-learn**: Simplifies model building, always preprocess and evaluate on test data.
- **Next Steps**: Explore ensembles, advanced tuning, and deep learning.

---

# Slide 13: Comparison of Classifiers
| Classifier          | Accuracy (Iris) | Strengths                     | Weaknesses                     |
|---------------------|----------------|-------------------------------|--------------------------------|
| Perceptron          | ~93%           | Fast, simple                  | Linear only, no probabilities   |
| Logistic Regression | ~97%           | Probabilities, interpretable  | Linear, sensitive to outliers  |
| Linear SVM          | High           | Robust, high-dimensional      | Computation, scaling needed