import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
df = pd.read_csv("dataset/pcb_defect_dataset_400_records.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDefect Distribution:")
print(df["Defect_Status"].value_counts())


# Defect distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Defect_Status")
plt.title("PCB Defect Distribution")
plt.xlabel("Defect Status")
plt.ylabel("Number of PCBs")
plt.show()



# Temperature vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Temperature_C")
plt.title("Temperature vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Temperature (°C)")
plt.show()


# Solder Quality vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Solder_Quality")
plt.title("Solder Quality vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Solder Quality")
plt.show()

# Component Alignment vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Component_Alignment")
plt.title("Component Alignment vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Component Alignment")
plt.show()


# Track Quality vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Track_Quality")
plt.title("Track Quality vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Track Quality")
plt.show()

# Voltage vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Voltage_V")
plt.title("Voltage vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Voltage (V)")
plt.show()


# Current vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Current_A")
plt.title("Current vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Current (A)")
plt.show()


# Resistance vs Defect Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Defect_Status", y="Resistance_Ohm")
plt.title("Resistance vs PCB Defect Status")
plt.xlabel("Defect Status")
plt.ylabel("Resistance (Ω)")
plt.show()

# Solder Type vs Defect Status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Solder_Type", hue="Defect_Status")
plt.title("Solder Type vs PCB Defect Status")
plt.xlabel("Solder Type")
plt.ylabel("Number of PCBs")
plt.show()


# Inspection Status vs Defect Status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Inspection_Status", hue="Defect_Status")
plt.title("Inspection Status vs PCB Defect Status")
plt.xlabel("Inspection Status")
plt.ylabel("Number of PCBs")
plt.show()

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

# Remove PCB_ID
df = df.drop("PCB_ID", axis=1)

# Separate features and target
X = df.drop("Defect_Status", axis=1)
y = df["Defect_Status"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())



#from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# Identify numerical and categorical columns
numerical_features = [
    "Voltage_V",
    "Current_A",
    "Resistance_Ohm",
    "Temperature_C",
    "Solder_Quality",
    "Component_Alignment",
    "Track_Quality"
]

categorical_features = [
    "Solder_Type",
    "Inspection_Status"
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same preprocessing to testing data
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)
# Convert target labels into numbers
y_train_encoded = y_train.map({
    "Non-Defective": 0,
    "Defective": 1
})

y_test_encoded = y_test.map({
    "Non-Defective": 0,
    "Defective": 1
})

print("\nEncoded Training Target:")
print(y_train_encoded.value_counts())

print("\nEncoded Testing Target:")
print(y_test_encoded.value_counts())
# -------------------------------
# LOGISTIC REGRESSION
# -------------------------------

model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

# Train the model
model.fit(X_train_processed, y_train_encoded)

print("\nLogistic Regression model trained successfully!")

# Make predictions on test data
y_pred = model.predict(X_test_processed)

print("\nPredictions:")
print(y_pred)

# -------------------------------
# MODEL EVALUATION
# -------------------------------

accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)

print("\nLogistic Regression Results")
print("----------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred,
    target_names=["Non-Defective", "Defective"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

decision_tree.fit(X_train_processed, y_train_encoded)

print("\nDecision Tree model trained successfully!")

# Prediction
y_pred_dt = decision_tree.predict(X_test_processed)

print("\nDecision Tree Predictions:")
print(y_pred_dt)

# Decision Tree Evaluation

accuracy_dt = accuracy_score(y_test_encoded, y_pred_dt)
precision_dt = precision_score(y_test_encoded, y_pred_dt)
recall_dt = recall_score(y_test_encoded, y_pred_dt)
f1_dt = f1_score(y_test_encoded, y_pred_dt)

print("\nDecision Tree Results")
print("----------------------------")
print("Accuracy :", accuracy_dt)
print("Precision:", precision_dt)
print("Recall   :", recall_dt)
print("F1 Score :", f1_dt)

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_dt,
    target_names=["Non-Defective", "Defective"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_dt))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

random_forest.fit(X_train_processed, y_train_encoded)

print("\nRandom Forest model trained successfully!")

# Prediction
y_pred_rf = random_forest.predict(X_test_processed)

print("\nRandom Forest Predictions:")
print(y_pred_rf)

# Random Forest Evaluation

accuracy_rf = accuracy_score(y_test_encoded, y_pred_rf)
precision_rf = precision_score(y_test_encoded, y_pred_rf)
recall_rf = recall_score(y_test_encoded, y_pred_rf)
f1_rf = f1_score(y_test_encoded, y_pred_rf)

print("\nRandom Forest Results")
print("----------------------------")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1 Score :", f1_rf)

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_rf,
    target_names=["Non-Defective", "Defective"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_rf))

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_processed, y_train_encoded)

print("\nKNN model trained successfully!")

# Prediction
y_pred_knn = knn.predict(X_test_processed)

print("\nKNN Predictions:")
print(y_pred_knn)

# KNN Evaluation

accuracy_knn = accuracy_score(y_test_encoded, y_pred_knn)
precision_knn = precision_score(y_test_encoded, y_pred_knn, zero_division=0)
recall_knn = recall_score(y_test_encoded, y_pred_knn, zero_division=0)
f1_knn = f1_score(y_test_encoded, y_pred_knn, zero_division=0)

print("\nKNN Results")
print("----------------------------")
print("Accuracy :", accuracy_knn)
print("Precision:", precision_knn)
print("Recall   :", recall_knn)
print("F1 Score :", f1_knn)

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_knn,
    target_names=["Non-Defective", "Defective"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_knn))

# Support Vector Machine (SVM)
from sklearn.svm import SVC

svm = SVC(
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train_processed, y_train_encoded)

print("\nSVM model trained successfully!")

# Prediction
y_pred_svm = svm.predict(X_test_processed)

print("\nSVM Predictions:")
print(y_pred_svm)

# SVM Evaluation

accuracy_svm = accuracy_score(y_test_encoded, y_pred_svm)
precision_svm = precision_score(
    y_test_encoded,
    y_pred_svm,
    zero_division=0
)
recall_svm = recall_score(
    y_test_encoded,
    y_pred_svm,
    zero_division=0
)
f1_svm = f1_score(
    y_test_encoded,
    y_pred_svm,
    zero_division=0
)

print("\nSVM Results")
print("----------------------------")
print("Accuracy :", accuracy_svm)
print("Precision:", precision_svm)
print("Recall   :", recall_svm)
print("F1 Score :", f1_svm)

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred_svm,
    target_names=["Non-Defective", "Defective"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_svm))

# ==========================================
# CROSS-VALIDATION
# ==========================================

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

# Create a new preprocessor for cross-validation
cv_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Logistic Regression Pipeline
logistic_pipeline = Pipeline([
    ("preprocessing", cv_preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# Decision Tree Pipeline
decision_tree_pipeline = Pipeline([
    ("preprocessing", cv_preprocessor),
    ("model", DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# Random Forest Pipeline
random_forest_pipeline = Pipeline([
    ("preprocessing", cv_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ))
])

# KNN Pipeline
knn_pipeline = Pipeline([
    ("preprocessing", cv_preprocessor),
    ("model", KNeighborsClassifier(
        n_neighbors=5
    ))
])

# SVM Pipeline
svm_pipeline = Pipeline([
    ("preprocessing", cv_preprocessor),
    ("model", SVC(
        class_weight="balanced",
        random_state=42
    ))
])

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline,
    "KNN": knn_pipeline,
    "SVM": svm_pipeline
}

# Cross-validation scoring
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

cv_results = []

for model_name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y.map({
            "Non-Defective": 0,
            "Defective": 1
        }),
        cv=cv,
        scoring=scoring
    )

    cv_results.append({
        "Model": model_name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean()
    })

cv_results_df = pd.DataFrame(cv_results)

print("\n5-Fold Cross-Validation Results")
print("--------------------------------")
print(cv_results_df.to_string(index=False))

# ==========================================
# MODEL COMPARISON
# ==========================================

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for metric in metrics:

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=cv_results_df,
        x="Model",
        y=metric
    )

    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Machine Learning Model")
    plt.ylabel(metric)
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.show()

    # ==========================================
# FINAL PCB DEFECT PREDICTION PIPELINE
# ==========================================

final_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

final_model = Pipeline([
    ("preprocessing", final_preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# Train final model using the complete dataset
final_model.fit(
    X,
    y.map({
        "Non-Defective": 0,
        "Defective": 1
    })
)



print("\nFinal PCB Defect Detection Model trained successfully!")
# ==========================================
# TEST NEW PCB
# ==========================================

new_pcb = pd.DataFrame([{
    "Voltage_V": 5.1,
    "Current_A": 0.8,
    "Resistance_Ohm": 6.2,
    "Temperature_C": 42.0,
    "Solder_Quality": 8,
    "Component_Alignment": 9,
    "Track_Quality": 8,
    "Solder_Type": "Lead-Free",
    "Inspection_Status": "Passed"
}])

prediction = final_model.predict(new_pcb)

if prediction[0] == 1:
    result = "Defective"
else:
    result = "Non-Defective"

print("\nNew PCB Prediction")
print("----------------------------")
print("Prediction:", result)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

import joblib

joblib.dump(final_model, "pcb_defect_model.pkl")

print("\nFinal model saved successfully!")