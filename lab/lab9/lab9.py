import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = fetch_olivetti_faces(shuffle=True, random_state=42)

X = data.images.reshape((data.images.shape[0], -1))  
y = data.target                                       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Naive Bayes Classifier Accuracy: {accuracy*100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()
for i in range(10):
    axes[i].imshow(X_test[i].reshape(64, 64), cmap='gray')
    axes[i].set_title(f"Pred: {y_pred[i]}\nActual: {y_test[i]}")
    axes[i].axis('off')
plt.tight_layout()
plt.savefig("lab9-img.png")