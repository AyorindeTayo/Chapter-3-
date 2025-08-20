# Chapter-3-
Machine Learning classifier using Scikitlearn 



# Machine Learning Classifiers Lab: Based on Chapter 3 of Python Machine Learning (2nd Edition)

## Overview
This lab is designed for students to explore the key concepts from Chapter 3: "A Tour of Machine Learning Classifiers Using scikit-learn" in *Python Machine Learning* (2nd Edition) by Sebastian Raschka and Vahid Mirjalili (Packt Publishing, 2017). The chapter introduces fundamental classification algorithms, their theoretical foundations, and practical implementation using the scikit-learn library. You'll work with the Iris dataset (for linear separability) and synthetic datasets like moons or XOR (for nonlinear problems) to train, evaluate, visualize, and tune models.

**Objectives:**
- Understand how to choose and implement classifiers (Perceptron, Logistic Regression, SVM, Kernel SVM, Decision Trees, Random Forests, KNN).
- Preprocess data, train models, and evaluate performance using metrics like accuracy.
- Visualize decision boundaries and tune hyperparameters to observe effects on model behavior.
- Compare classifiers and discuss strengths/weaknesses based on data characteristics (e.g., linear vs. nonlinear separability, overfitting).

**Prerequisites:**
- Python 3.x with scikit-learn, NumPy, Pandas, Matplotlib (install via `pip install scikit-learn numpy pandas matplotlib`).
- Familiarity with Chapter 3 concepts (e.g., No Free Lunch theorem, regularization, kernels).
- Estimated Time: 3-4 hours.
- Work in a Jupyter Notebook for easy execution and visualization.

**Dataset Notes:**
- Primary: Iris (built-in via `sklearn.datasets.load_iris`).
- Nonlinear: Use `make_moons` or `make_xor` from `sklearn.datasets`.
- Always split data (70/30 train/test), standardize features where needed (e.g., for SVM, KNN).

---

## Helper Functions

Use these for visualizations (adapted from the book's code repository).

### Plot Decision Regions
```python
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=100, label='test set')
    plt.legend(loc='upper left')
    plt.show()
```

---

## Exercise 1: Data Preparation and Choosing a Classifier

**Concepts Covered:** Preprocessing, No Free Lunch theorem, workflow for classifier selection.

**Steps:**
1. Load the Iris dataset and select features (e.g., petal length and width for 2D visualization).
2. Split into train/test sets and standardize.
3. Discuss: Why no single classifier is best? Consider dataset size, noise, separability.

**Code Template:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length and width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

print("Class distribution:", np.bincount(y_train))
```

**Tasks:**
- Print class distributions (`np.bincount(y_train)`).
- Generate a nonlinear dataset using `make_moons`. Preprocess similarly.

---

## Exercise 2: Perceptron

**Concepts Covered:** Linear classification, convergence on separable data.

**Code Template:**
```python
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))

plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
```

**Tasks:**
- Vary `eta0` (0.01, 0.1, 1). Observe convergence/misclassifications.
- Try on moons dataset. Why does it fail?

---

## Exercise 3: Logistic Regression

**Concepts Covered:** Probability modeling, regularization (L2 via C).

**Code Template:**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Probabilities:', lr.predict_proba(X_test_std[:3]))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
```

**Tasks:**
- Tune `C` (0.01, 1, 100). Plot boundaries and discuss overfitting/underfitting.
- Implement from scratch (modify Adaline if time allows).

---

## Exercise 4: Support Vector Machines (SVM)

**Concepts Covered:** Maximum margin, soft margins via C.

**Code Template:**
```python
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
```

**Tasks:**
- Increase `C` to 100. How does the margin change?
- Identify support vectors (`svm.support_vectors_`).

---

## Exercise 5: Kernel SVM

**Concepts Covered:** Kernel trick for nonlinear data (RBF kernel, gamma).

**Code Template (Moons Dataset):**
```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=100, random_state=123)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moons, y_moons, test_size=0.3, random_state=1)

sc_m = StandardScaler()
X_train_m_std = sc_m.fit_transform(X_train_m)
X_test_m_std = sc_m.transform(X_test_m)
X_combined_m_std = np.vstack((X_train_m_std, X_test_m_std))
y_combined_m = np.hstack((y_train_m, y_test_m))

svm_rbf = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm_rbf.fit(X_train_m_std, y_train_m)

y_pred_m = svm_rbf.predict(X_test_m_std)
print('Accuracy:', accuracy_score(y_test_m, y_pred_m))

plot_decision_regions(X_combined_m_std, y_combined_m, classifier=svm_rbf, test_idx=range(len(X_train_m), len(X_combined_m_std)))
plt.xlabel('Feature 1 [standardized]')
plt.ylabel('Feature 2 [standardized]')
```

**Tasks:**
- Tune `gamma` (0.01, 1, 100). Observe overfitting.
- Apply to Iris and compare with linear SVM.

---

## Exercise 6: Decision Trees

**Concepts Covered:** Information gain (Gini/entropy), pruning.

**Code Template:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)  # No scaling needed

y_pred = tree.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

X_combined = np.vstack((X_train, X_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(len(X_train), len(X_combined)))
plt.xlabel('Petal length')
plt.ylabel('Petal width')

dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginica'],
                          feature_names=['petal length', 'petal width'], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
```

**Tasks:**
- Vary `max_depth` (1–10). Export and interpret the tree.
- Use entropy criterion; compare accuracy.

---

## Exercise 7: Random Forests

**Concepts Covered:** Ensemble learning, bagging, feature importance.

**Code Template:**
```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Feature Importances:', forest.feature_importances_)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(len(X_train), len(X_combined)))
plt.xlabel('Petal length')
plt.ylabel('Petal width')
```

**Tasks:**
- Increase `n_estimators` to 100. Plot feature importances.
- Use OOB score (`oob_score=True`).

---

## Exercise 8: K-Nearest Neighbors (KNN)

**Concepts Covered:** Lazy learning, distance metrics, bias-variance tradeoff.

**Code Template:**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

y_pred = knn.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
```

**Tasks:**
- Vary `n_neighbors` (1, 5, 10). Discuss under/overfitting.
- Use Manhattan distance (`p=1`).

---

## Exercise 9: Hyperparameter Tuning and Comparison

**Concepts Covered:** Grid search, cross-validation, model selection.

**Code Template:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_std, y_train)
print('Best params:', grid.best_params_)
print('Best score:', grid.best_score_)

models = {'Perceptron': ppn, 'LogReg': lr, 'SVM': svm, 'Tree': tree, 'Forest': forest, 'KNN': knn}
for name, model in models.items():
    y_pred = model.predict(X_test_std if name in ['Perceptron', 'LogReg', 'SVM', 'KNN'] else X_test)
    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred)}')
```

**Tasks:**
- Apply `GridSearchCV` to Logistic Regression and KNN.
- Create a table comparing accuracies across classifiers on Iris and moons.
- Discuss: Which works best for linear vs. nonlinear data?

---

## Conclusion and Analysis

- Save your notebook with all plots, accuracies, and discussions and push to Github repositery.
- Key Questions:
  - How does regularization prevent overfitting?
  - When to use ensembles vs. simple models?


**Resources:** Book's GitHub repo for full code.


