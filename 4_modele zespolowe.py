import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# 1. Pobieranie danych
CSV_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(CSV_URL, header=None, names=COLS)

print("Wczytano dane — rozmiar (wiersze, kolumny):", df.shape)
print(df.head())

# 2. ANaliza

print(df.describe().T.round(2).to_string())

cols_zero_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_zero_missing] = df[cols_zero_missing].replace(0, np.nan)
for c in cols_zero_missing:
    med = df[c].median()
    df[c] = df[c].fillna(med)

print("Braki danych (liczba NaN) po uzupełnieniu:", df.isna().sum().sum())

# 3. Przygotowanie cech i etykiet
X = df.drop(columns="Outcome")
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# 4. Modele do porównania
# - RandomForest - dobrze sobie radzi z szumem i interakcjami
# - GradientBoosting - silny model, dostosowuje się iteracyjnie

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1
    ),
"GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,  # mniejszy learning_rate -> często lepsze, wolniejsze uczenie
        random_state=42
    )
}
results = {}
# 5. Trening

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

    print(f"Wyniki dla {name}:")
    print(f" accuracy:  {acc:.4f}")
    print(f" precision: {prec:.4f}")
    print(f" recall:    {rec:.4f}")
    print(f" f1:        {f1:.4f}")
    print(f" ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification report (szczegóły dla każdej klasy):")
    print(classification_report(y_test, y_pred, digits=4))



