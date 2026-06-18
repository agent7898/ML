import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42) 
x_values = np.random.rand(100, 1)  
y_labels = np.array([1 if x <= 0.5 else 2 for x in x_values.flatten()]) 

X_train = x_values[:50]   
y_train = y_labels[:50]  
X_test = x_values[50:]    
y_test = y_labels[50:]    

k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

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