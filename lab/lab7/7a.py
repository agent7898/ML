import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml

boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data
y = boston.target


X_train, X_test, y_train, y_test = train_test_split(X[['RM']], y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

acc = mean_squared_error(y_test,y_pred)
print(f"MSE={acc}")


plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.5)
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Linear Regression")
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.legend()
plt.title("Linear Regression on Boston Housing Dataset")
plt.savefig("lr.png")