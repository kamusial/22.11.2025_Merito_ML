import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print(df.head(10))

print("\nInformacje o danych:")
print(df.info())
print("\nStatystyki liczbowe: ")
print(df.describe())

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("\n" + "="*50)
print("1. Proste modele klasyfikacji")
print("="*50)

print('\nWczytuje dane medyczne - rak piersi')
data = load_breast_cancer()
X = data.data
y = data.target

print(f"Wczytano dane: {X.shape[0]} pacjentów, {X.shape[1]} pomiarów")
print(f"Diagnozy: {np.sum(y==0)} chorych, {np.sum(y==1)} zdrowych")
print("\nDziele dane na zbior trenignowy i testowy")
#skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Dane treningowe: {X_train.shape[0]} próbek")
print(f"Dane testowe: {X_test.shape[0]} próbek")
print("Dane zostały przeskalowane")

print("\nTrenuje różne modele klasyfikacji")
#model 1 regresja logistyczna
print("1. uczę Regresję Logistyczną")
log_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

print("2. Uczę Random Forest")
rf_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) #dokladnosc modelu random forest

print("3. Uczę KNN")
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("\nPorównanie dokładności modeli:")
print("=" * 40)
print(f"1. Regresja Logistyczna: {accuracy_log:.4f} ({accuracy_log*100:.1f}%)")
print(f"2. Random Forest: {accuracy_rf:.4f} ({accuracy_rf*100:.1f}%)")
print(f"3. KNN: {accuracy_knn:.4f} ({accuracy_knn*100:.1f}%)")

models = ['Regresja\nLogistczna', 'Random\nForest', 'KNN']
accuracies = [accuracy_log, accuracy_rf, accuracy_knn]
plt.figure(figsize=(10, 6))
colors = ['lightblue', 'lightgreen', 'lightcoral']
bars = plt.bar(models, accuracies, color=colors)
for bar, accuracy in zip (bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height()+0.005,
        f'{accuracy:.3f}',
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.ylabel("Dokładność (Accuracy)")
plt.title("Porównanie Dokładności Modeli Klasyfikacji")
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

#znajdowanie najlepszego modelu na podstawie dokładności
best_accuracy = max(accuracies)
best_model_idx = accuracies.index(max(accuracies))
best_model_name = models[best_model_idx]

print(f"\nNajlepszy model: {best_model_name} - {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")

#analiza najlepszego modelu
print(f"\nAnaliza najlepszego modelu: {best_model_name}")
if best_model_name == 'Regresja\nLogistczna':
    best_model = log_model
    y_pred_best = y_pred_log
elif best_model_name == 'Random Forest':
    best_model = rf_model
    y_pred_best = y_pred_rf
else:
    best_model = knn_model
    y_pred_best = knn_model

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Macierz pomyłek - {best_model_name}")
plt.ylabel('Prawdziwa diagnoza')
plt.xlabel('Przewidziana diagnoza')
plt.tight_layout()
plt.show()

print(f"\nRaport dla {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=data.target_names))

print("\n" + '='*40)
print("Regresja Domy w Californi")
print("="*40)

print("\nWczytuję dane o cenach domów")
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score #metryki jakosci dla regresji

housing = fetch_california_housing()
X_housing = housing.data #cechy domów
y_housing = housing.target #ceny domów

print(f"wczytano: {X_housing.shape[0]} domów, {X_housing.shape[1]} cech")
print(f"średnia cena domu: ${y_housing.mean()*100000:.0f}")

scaler_reg = StandardScaler()
X_housing_scaled = scaler_reg.fit_transform(X_housing)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing_scaled,
    y_housing,
    test_size=0.3,
    random_state=42,
)

print("\nTrenuję model regresji")
lin_model = LinearRegression()
lin_model.fit(X_train_h, y_train_h)
y_pred_lin = lin_model.predict(X_test_h)

#ocena modelu regresji
mse = mean_squared_error(y_test_h, y_pred_lin) #średni błąd kwadratowy - im mniejszy tym lepiej
r2 = r2_score(y_test_h, y_pred_lin) #im blizej 1 tym lepiej, im blizej 0 to gorzej

print("\nWyniki Regresji:")
print(f"Średni błąd kwadratowy (mse): {mse:.4f}")
print(f"Współczynnik determinacji: {r2:.4f}")

plt.figure(figsize = (8, 6))
plt.scatter(y_test_h, y_pred_lin, alpha = 0.5)
plt.plot(
    [y_test_h.min(), y_test_h.max()],
    [y_test_h.min(), y_test_h.max()],
    'r--',
    lw=2
)

plt.xlabel("Rzeczywiste ceny")
plt.ylabel("Przewidywane ceny")
plt.title("Regresja Liniowa: Rzeczywiste vs Przewidywane ceny domów")
plt.tight_layout()
plt.show()