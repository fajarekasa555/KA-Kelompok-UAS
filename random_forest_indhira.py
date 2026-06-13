# ============================================================
# UAS KECERDASAN ARTIFISIAL
# RANDOM FOREST
# Nama  : Indhira Yuantika Christy
# NIM   : 244107020171
# Studi Kasus: Prediksi Penyakit Jantung
# ============================================================

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Membuat folder output secara otomatis jika belum ada
os.makedirs("output", exist_ok=True)


# ============================================================
# 1. MEMBACA DATASET
# ============================================================

df = pd.read_csv("data/heart.csv")

print("Lima data pertama:")
print(df.head())

print("\nJumlah baris dan kolom:", df.shape)
print("Jumlah baris:", df.shape[0])
print("Jumlah kolom:", df.shape[1])

print("\nInformasi dataset:")
print(df.info())

print("\nStatistik deskriptif:")
print(df.describe().T)


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Mengecek missing value
missing_values = df.isnull().sum()

missing_df = pd.DataFrame({
    "Kolom": missing_values.index,
    "Jumlah Missing Value": missing_values.values
})

print("\nJumlah Missing Value pada Setiap Kolom:")
print(missing_df)

# Mengecek data duplikat
duplicate_count = df.duplicated().sum()
print("\nJumlah data duplikat:", duplicate_count)

# Menghapus data duplikat jika ada
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Data duplikat berhasil dihapus.")
else:
    print("Tidak terdapat data duplikat.")

print("Ukuran dataset setelah cleaning:", df.shape)


# ============================================================
# 3. VISUALISASI DATA
# ============================================================

# Distribusi target
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="target")
plt.title("Distribusi Target Penyakit Jantung")
plt.xlabel("Target: 0 = Tidak Sakit, 1 = Sakit")
plt.ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig("output/distribusi_target.png")
plt.show()

# Distribusi fitur
df.hist(figsize=(10, 10), bins=20)
plt.suptitle("Distribusi Fitur pada Dataset Heart Disease", fontsize=16)
plt.tight_layout()
plt.savefig("output/distribusi_fitur.png")
plt.show()

# Heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Antar Fitur")
plt.tight_layout()
plt.savefig("output/heatmap_korelasi.png")
plt.show()

# Korelasi fitur terhadap target
target_corr = df.corr()["target"].sort_values(ascending=False)

plt.figure(figsize=(8, 6))
target_corr.drop("target").sort_values().plot(kind="barh")
plt.title("Korelasi Fitur terhadap Target")
plt.xlabel("Nilai Korelasi")
plt.ylabel("Fitur")
plt.tight_layout()
plt.savefig("output/korelasi_fitur_target.png")
plt.show()


# ============================================================
# 4. DATA PREPARATION
# ============================================================

# Memisahkan fitur dan target
X = df.drop("target", axis=1)
y = df["target"]

print("\nUkuran fitur X:", X.shape)
print("Ukuran target y:", y.shape)

# Mengecek kolom kategorikal
categorical_cols = X.select_dtypes(include=["object"]).columns
print("Kolom kategorikal:", list(categorical_cols))

# Encoding jika terdapat kolom kategorikal
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print("Encoding data kategorikal selesai.")
else:
    print("Tidak terdapat kolom kategorikal, encoding tidak diperlukan.")

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah data training:", X_train.shape[0])
print("Jumlah data testing:", X_test.shape[0])


# ============================================================
# 5. IMPLEMENTASI RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nHasil prediksi Random Forest:")
print(y_pred_rf)


# ============================================================
# 6. EVALUASI RANDOM FOREST
# ============================================================

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nEvaluasi Model Random Forest")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1-Score :", f1_rf)

eval_rf = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy_rf],
    "Precision": [precision_rf],
    "Recall": [recall_rf],
    "F1-Score": [f1_rf]
})

print("\nTabel Evaluasi Random Forest:")
print(eval_rf)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report Random Forest:")
print(classification_report(y_test, y_pred_rf))


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nConfusion Matrix Random Forest:")
print(cm_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("output/confusion_matrix_random_forest.png")
plt.show()


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance Random Forest:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Fitur"
)

plt.title("Feature Importance pada Model Random Forest")
plt.xlabel("Nilai Importance")
plt.ylabel("Fitur")
plt.tight_layout()
plt.savefig("output/feature_importance_random_forest.png")
plt.show()


# ============================================================
# 10. INTERPRETASI SINGKAT
# ============================================================

print("\nInterpretasi:")
print("Model Random Forest digunakan untuk memprediksi penyakit jantung berdasarkan fitur medis pasien.")
print("Evaluasi dilakukan menggunakan accuracy, precision, recall, dan F1-score.")
print("Confusion matrix digunakan untuk mengetahui jumlah prediksi benar dan salah.")
print("Feature importance digunakan untuk mengetahui fitur yang paling berpengaruh terhadap hasil prediksi model.")