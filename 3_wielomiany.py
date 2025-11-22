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

    re