from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1) Generowanie danych
np.random.seed(42)
n_samples = 100
X = np.linspace(-3, 3, n_samples).reshape(-1, 1)  #z jednej tablicy 1D do macierzy 2D z jedną kolumną

# funkcja nieliniowa (suma wielomianu i sinus)
y_true = 0.5 * X.ravel()**3 - X.ravel()**2 + 2.0 * np.sin(1.5 * X.ravel())
# dodamy szum
y = y_true + np.random.normal(scale=3.0, size=n_samples)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

def evaluate(name, model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test, y_pred_test)

    print(f'{name}:')
    print(f' MSE train = {mse_train:.2f}, MSE test = {mse_test:.2f}')
    print(f"  R2  train = {r2_train:.2f},  R2  test = {r2_test:.2f}\n")

# regresja liniowa
linear = LinearRegression()
linear.fit(X_train, y_train)
evaluate("Linear Regression", linear, X_train, X_test, y_train, y_test)

# regresja wielomianowa
degree = 9 # wysokie ryzyko przeuczenia
poly = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
poly.fit(X_train, y_train)
evaluate(f"Polynomial deg={degree}", poly, X_train, X_test, y_train, y_test)

# RIDGE i LASSO - regularyzacja L2 i L1
alpha = 1.0 # siła regularyzacji

ridge = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=alpha))
ridge.fit(X_train, y_train)
evaluate(f"Ridge deg={degree}", ridge, X_train, X_test, y_train, y_test)

lasso = make_pipeline(PolynomialFeatures(degree, include_bias=False), Lasso(alpha=alpha, max_iter=5000))
lasso.fit(X_train, y_train)
evaluate(f"Lasso deg={degree}", lasso, X_train, X_test, y_train, y_test)

# wykresy - porównanie modeli
X_plot = np.linspace(-3, 3, 400).reshape(-1, 1)
plt.figure(figsize=(11, 7))

# dane
plt.scatter(X_train, y_train, color='black', s=20, label='Train')
plt.scatter(X_test, y_test, color='grey', s=20, label='Test', alpha=0.7)

# przewidywanie modeli
plt.plot(X_plot, linear.predict(X_plot), label="Linear")
plt.plot(X_plot, poly.predict(X_plot), label=f"Poly deg={degree}")
plt.plot(X_plot, ridge.predict(X_plot), label=f"Ridge α={alpha}")
plt.plot(X_plot, lasso.predict(X_plot), label=f"Lasso α={alpha}")

plt.title("Porównanie: Linear, Polynomial, Ridge, Lasso")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


