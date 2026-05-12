import matplotlib
matplotlib.use('Agg')  # Use the TkAgg backend for stable display

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate 100 random values of x in the range [0,1]
np.random.seed(42)  # For reproducibility
x_values = np.random.rand(100, 1)  # 100 random values in the range [0,1]
# Step 2: Label the first 50 points as Class1 and the rest as Class2
y_labels = np.array([1 if x <= 0.5 else 2 for x in x_values.flatten()])  # Class1: x ≤ 0.5, Class2: x > 0.5

# Split the training and test sets
X_train = x_values[:50]   # First 50 points
y_train = y_labels[:50]   # First 50 labels
X_test = x_values[50:]    # Remaining 50 points
y_test = y_labels[50:]    # Remaining 50 labels

# Step 3: Classify using KNN for different k values
k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    # Initialize the KNN classifier with the current k value
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = knn.predict(X_test)

    # Plot the decision boundary and the points
    plt.figure()
    plt.scatter(X_test, y_test, c='blue', s=30, label='True Label')
    plt.scatter(X_test, y_pred, c='red', s = 10, marker='x', label='Predicted Label')
    plt.title(f'KNN Classification with k={k}')
    plt.xlabel('x')
    plt.ylabel('Class Label')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"KNN_Classification for k={k}.png")
    accuracy = knn.score(X_test, y_test)
    print(f"Accuracy for k={k}: {accuracy:.2f}")