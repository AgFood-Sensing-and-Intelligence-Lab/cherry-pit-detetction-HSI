

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random
print(random.choice([0, 90])) # To random select orientation either 0 degree or 90 degree
random_state = 42
# ------------------ Configuration ------------------
train0_path = "Training set\Training_0.xlsx"
train90_path = "Training set\Training_90.xlsx"
test_path   = "Test set\Testing_90.xlsx" 

# Wavelengths/sepctra region wavelengths
selected_wavelengths = [1029, 1032, 1036]  # WS3-RI-1

# -------- Read & Prepare Data ----------------------
train0 = pd.read_excel(train0_path)
train90 = pd.read_excel(train90_path)
train = pd.concat([train0, train90], ignore_index=True)
train = train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

test = pd.read_excel(test_path)
test = test.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# -------- Select Only Specific Wavelengths ---------
# Convert wavelength numbers to string to match Excel headers
available_cols = [str(c) for c in train.columns[1:]]  # skip label column
selected_cols = [str(w) for w in selected_wavelengths if str(w) in available_cols]

if not selected_cols:
    raise ValueError("❌ None of the specified wavelengths were found in the Excel file!")

X_train = train[selected_cols].values
y_train = train.iloc[:, 0].astype(int).values

X_test = test[selected_cols].values
y_test = test.iloc[:, 0].astype(int).values

print(f"✅ Using wavelengths: {selected_cols}")
#print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# -------- Helper Function for Evaluation -----------
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    def report(split_name, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n{name} — {split_name}")
        print("Confusion matrix (rows=true [0,1]; cols=pred [0,1]):")
        print(cm)
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

    ytr_pred = model.predict(X_tr)
    yte_pred = model.predict(X_te)
    report("TRAIN", y_tr, ytr_pred)
    report("TEST", y_te, yte_pred)

# -------- Cross-Validation Setup -------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# ----------- SVM (RBF) -----------------------------
svm_pipe = Pipeline([("clf", SVC())])

#    ("scaler", StandardScaler()),

svm_grid = {
    "clf__kernel": ["rbf", "linear"],
    "clf__C": [0.1, 1, 10],
    "clf__gamma": [ 0.01, 0.1, 1],}

svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv, n_jobs=-1, scoring="f1", refit=True, verbose=0)
svm_search.fit(X_train, y_train)
print("\nBest SVM params:", svm_search.best_params_)
svm_best = svm_search.best_estimator_

# ----------- Random Forest -------------------------
rf_pipe = Pipeline([("clf", RandomForestClassifier(random_state=random_state))])

rf_grid = {
    "clf__n_estimators": [5, 10,20,50],
    "clf__max_depth": [5, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],}

rf_search = GridSearchCV(rf_pipe, rf_grid, cv=cv, n_jobs=-1, scoring="f1", refit=True, verbose=0)
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
rf_best = rf_search.best_estimator_

# --------------- Evaluation ------------------------
evaluate_model("SVM (RBF)", svm_best, X_train, y_train, X_test, y_test)
evaluate_model("Random Forest", rf_best, X_train, y_train, X_test, y_test)
