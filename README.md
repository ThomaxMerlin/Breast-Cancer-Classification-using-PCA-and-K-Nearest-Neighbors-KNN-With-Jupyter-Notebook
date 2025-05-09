Breast Cancer Classification with PCA and KNN
Overview
GitHub
+2
GitHub
+2
GitHub
+2
This project utilizes Principal Component Analysis (PCA) for dimensionality reduction and K-Nearest Neighbors (KNN) for classifying breast cancer data. The dataset employed is the Breast Cancer Wisconsin (Diagnostic) Dataset from scikit-learn.

Prerequisites
Python 3.x

Required libraries:
GitHub

numpy

pandas

scikit-learn

matplotlib
GitHub
+2
GitHub
+2
GitHub
+2

jupyter
GitHub
+8
GitHub
+8
GitHub
+8

Install dependencies:
GitHub

bash
Copy
Edit
pip install numpy pandas scikit-learn matplotlib jupyter
Execution
GitHub
+5
GitHub
+5
GitHub
+5
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open the .ipynb file.

Run all cells sequentially.

Methodology
GitHub
+8
GitHub
+8
GitHub
+8
Data Loading
GitHub
+7
GitHub
+7
GitHub
+7

python
Copy
Edit
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
Data Standardization

python
Copy
Edit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)
PCA Application

python
Copy
Edit
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
KNN Classification
GitHub
+1
GitHub
+1

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled, cancer.target, test_size=0.2, random_state=0)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
PCA Variance Analysis

python
Copy
Edit
pca_full = PCA().fit(cancer.data)
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
Results
KNN model accuracy: Approximately 93.86%

First two principal components capture ~99.82% variance

License
MIT License
