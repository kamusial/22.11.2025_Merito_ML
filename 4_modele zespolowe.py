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

print(df.describe().T.round(2).to_string())

cols_zero_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_zero_missing] = df[cols_zero_missing].replace(0, np.nan)
for c in cols_zero_missing:
    med = df[c].median()
    df[c] = df[c].fillna(med)

print("Braki danych (liczba NaN) po uzupełnieniu:", df.isna().sum().sum())





