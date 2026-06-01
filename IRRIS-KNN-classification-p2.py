# ============================================================
#  DecodeLabs | AI Internship 2026
#  Project 2: Data Classification Using AI
#  Dataset: Iris | Algorithm: K-Nearest Neighbors (KNN)
# ===========================================================
# STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
#STEP 2: Load & Understand the Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("=" * 55)
print("       DecodeLabs — Project 2: Iris Classification")
print("=" * 55)
print(f"\nDataset Shape  : {df.shape}")
print(f" Classes        : {list(iris.target_names)}")
print(f"Features       : {iris.feature_names}\n")
print("First 5 Rows ")
print(df.head())
print("\nClass Distribution ")
print(df['species'].value_counts())
print("\nBasic Statistics")
print(df.describe())
# STEP 3: Feature Scaling (Gatekeeper Rule)
X = iris.data
y = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeature Scaling applied (StandardScaler: Mean=0, Variance=1)")
# STEP 4: Train-Test Split (80/20 + Shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
#  STEP 5: Find Optimal K (Elbow Method)
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))
optimal_k = k_range[np.argmin(error_rates)]
print(f"\nOptimal K found : {optimal_k}")
#  STEP 6: Train the KNN Model
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)                  # FIT  Memorize the map
predictions = model.predict(X_test)          # PREDICT Apply logic
# STEP 7: Output Validation
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)
print("\n" + "=" * 55)
print("            MODEL RESULTS")
print("=" * 55)
print(f"  Accuracy  : {accuracy * 100:.2f}%")
print(f"  F1 Score  : {f1:.4f}")
print("\n Classification Report ")
print(classification_report(y_test, predictions, target_names=iris.target_names))
print("Confusion Matrix ")
print(cm)
# STEP 8: Visualizations 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("DecodeLabs | Project 2 — Iris KNN Classification", fontsize=14, fontweight='bold')
# --- Plot 1: Elbow Curve (Optimal K) ---
axes[0].plot(k_range, error_rates, marker='o', color='steelblue', linewidth=2)
axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axes[0].set_title("Elbow Curve — Choosing K")
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Error Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# --- Plot 2: Confusion Matrix ---
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=axes[1],)
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
# --- Plot 3: Feature Distribution (Petal Length vs Petal Width) ---
colors =['#E74C3C', '#3498DB', '#2ECC71']
for i, (species, color) in enumerate(zip(iris.target_names, colors)):
    mask = y == i
    axes[2].scatter(
        X[mask, 2], X[mask, 3],
        label=species, color=color, alpha=0.7, edgecolors='k', linewidths=0.5)
axes[2].set_title("Petal Length vs Petal Width")
axes[2].set_xlabel("Petal Length (cm)")
axes[2].set_ylabel("Petal Width (cm)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\admin\Downloads\iris_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n Visualization saved as 'iris_results.png'")
# STEP 9: Test with Custom Input
print("\n" + "=" * 55)
print("         Custom Prediction (Sample Test)")
print("=" * 55)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])        # Likely Setosa
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
proba = model.predict_proba(sample_scaled)[0]
print(f"  Input   : Sepal L=5.1, Sepal W=3.5, Petal L=1.4, Petal W=0.2")
print(f"  Predicted Species : {iris.target_names[result[0]].upper()}")
print(f"  Confidence Setosa: {proba[0]*100:.1f}% | Versicolor: {proba[1]*100:.1f}% | Virginica: {proba[2]*100:.1f}%")
print("\n Project 2 Complete! Badge Earned ")