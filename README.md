Breast Cancer Classification using PCA and K-Nearest Neighbors (KNN)
With Jupyter Notebook

This Jupyter Notebook demonstrates how to perform breast cancer classification using Principal Component Analysis (PCA) for dimensionality reduction and K-Nearest Neighbors (KNN) for classification. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from scikit-learn.

Table of Contents
Prerequisites

Getting Started

Running the Code

Code Explanation

Results

License

Prerequisites
Before running the code, ensure you have the following installed:

Python 3.x

Required Python libraries:

bash
Copy
pip install numpy pandas scikit-learn matplotlib jupyter
Jupyter Notebook (to run the .ipynb file).

Getting Started
Launch Jupyter Notebook
Start Jupyter Notebook:

bash
Copy
jupyter notebook
Open the .ipynb file from the Jupyter Notebook interface.

Running the Code
Open the .ipynb file in Jupyter Notebook.

Run each cell sequentially to execute the code.

Code Explanation
1. Import Libraries
python
Copy
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Libraries used for data loading, preprocessing, visualization, and modeling.

2. Load and Explore Data
python
Copy
cancer = load_breast_cancer()
print("Shape of data:", cancer.data.shape)
print("Target names:", cancer.target_names)
Load the Breast Cancer dataset and explore its structure.

3. Data Visualization
python
Copy
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()
for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color='red', alpha=.5, label='Malignant')
    ax[i].hist(benign[:, i], bins=bins, color='blue', alpha=.5, label='Benign')
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature Magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(loc="best")
fig.tight_layout()
Visualize the distribution of each feature for malignant and benign cases.

4. Standardize Data
python
Copy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)
Standardize the features to have zero mean and unit variance.

5. Apply PCA
python
Copy
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Original Shape:", X_scaled.shape)
print("Reduced Shape:", X_pca.shape)
Reduce the dimensionality of the dataset to 2 principal components.

6. Visualize PCA Results
python
Copy
plt.figure(figsize=(8, 8))
for label in np.unique(cancer.target):
    plt.scatter(X_pca[cancer.target == label, 0], X_pca[cancer.target == label, 1], label=cancer.target_names[label])
plt.legend()
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
Visualize the data in the reduced 2D space.

7. Train KNN Model
python
Copy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, cancer.target, test_size=0.2, random_state=0)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
Train a KNN model on the scaled data and evaluate its accuracy.

8. Evaluate PCA Components
python
Copy
pca = PCA().fit(cancer.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
print("Variance Captured by First 2 Principal Components:", pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
print("Variance Explained by 30 Components:", sum(pca.explained_variance_ratio_))
Analyze the cumulative explained variance by PCA components.

Results
Accuracy: The KNN model achieves an accuracy of 93.86% on the test set.

PCA Visualization: The first two principal components capture 99.82% of the variance.

Cumulative Explained Variance: The plot shows how much variance is explained by each principal component.

License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

Support
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at minthukywe2020@gmail.com.

Enjoy exploring breast cancer classification using PCA and KNN in Jupyter Notebook! 🚀
