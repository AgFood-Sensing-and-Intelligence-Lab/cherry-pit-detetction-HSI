import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import random
print(random.choice([0, 90])) # To random select orientation either 0 degree or 90 degree
random_state = 42


train0_path = "Training set\Training_0.xlsx"
train90_path = "Training set\Training_90.xlsx"
test_path   = "Test set\Testing_90.xlsx"

train0 = pd.read_excel(train0_path)
train90 = pd.read_excel(train90_path)

train = pd.concat([train0, train90], ignore_index=True)
train_df = train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)  # shuffle


test = pd.read_excel(test_path)
test_df = test.sample(frac=1.0, random_state=random_state).reset_index(drop=True)    # shuffle

# First column = Label, remaining columns = Spectra
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values

y_test = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine optimal number of latent variables
LV_range = range(2, 15)  
mean_scores = []

for lv in LV_range:
    pls = PLSRegression(n_components=lv)
    X_lv = pls.fit_transform(X_train_scaled, y_train)[0]
    
    svm = SVC(kernel='rbf')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(svm, X_lv, y_train, cv=cv, scoring='accuracy')
    mean_scores.append(scores.mean())

# Display results
for lv, score in zip(LV_range, mean_scores):
    print(f"LVs: {lv}, CV Accuracy: {score:.3f}")

best_lv = LV_range[np.argmax(mean_scores)]
print(f"\nOptimal number of LVs: {best_lv}")


#PLS-DA for Feature Extraction

pls = PLSRegression(n_components=6) # put here optimal no. of LVs
pls.fit(X_train_scaled, y_train)

# Transform data into latent variable space
X_train_pls = pls.transform(X_train_scaled)
X_test_pls = pls.transform(X_test_scaled)


# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Optimize SVM 
svm_param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
    "kernel": ["rbf", "linear"]
}

svm_grid = GridSearchCV(
    SVC(probability=True),
    svm_param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

svm_grid.fit(X_train_pls, y_train)
best_svm = svm_grid.best_estimator_

print("\nBest SVM Parameters:", svm_grid.best_params_)

# Optimize Random Forest 
rf_param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

rf_grid.fit(X_train_pls, y_train)
best_rf = rf_grid.best_estimator_

print("\nBest RF Parameters:", rf_grid.best_params_)

# Final optimized SVM model
svm_clf = best_svm
svm_clf.fit(X_train_pls, y_train)
y_pred_svm = svm_clf.predict(X_test_pls)

# Final optimized Random Forest model
rf_clf = best_rf
rf_clf.fit(X_train_pls, y_train)
y_pred_rf = rf_clf.predict(X_test_pls)



print("\n===== Optimized SVM PERFORMANCE =====")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

print("\n===== Optimized RF PERFORMANCE =====")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

