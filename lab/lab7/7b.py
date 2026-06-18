import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

auto_mpg = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")
auto_mpg.dropna(inplace=True)  

auto_mpg = auto_mpg[auto_mpg['horsepower'] != '?']

X_auto = auto_mpg[['horsepower']]
y_auto = auto_mpg['mpg']

X_train, X_test, y_train, y_test = train_test_split(X_auto, y_auto, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

mse=mean_squared_error(y_test, y_pred_poly)
print (f"mse: {mse}")

plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.5)
plt.scatter(X_test, y_pred_poly, color="red", label="Predicted Data", alpha=0.5)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.title("Polynomial Regression on Auto MPG Dataset")
plt.savefig("poly.png")
plt.close()