
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_excel(r"D:\243project\Flipkart Sales Dataset.xlsx")

sns.set_theme(style="whitegrid")

print("\n========== DATASET OVERVIEW ==========")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

# ==========================================================
# 2. MISSING VALUES ANALYSIS
# ==========================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])
print("Total Missing Values:", df.isnull().sum().sum())

# ==========================================================
# 3. DUPLICATE RECORDS
# ==========================================================

print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate Rows:", df.duplicated().sum())

# ==========================================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================================

numerical_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

# ==========================================================
# 5. NUMERICAL SUMMARY
# ==========================================================

print("\n========== NUMERICAL SUMMARY ==========")

print(df.describe().T.round(2))

# ==========================================================
# 6. CATEGORICAL SUMMARY
# ==========================================================

print("\n========== CATEGORICAL SUMMARY ==========")

for col in categorical_columns:
    print("\nColumn:", col)
    print("Unique Values:", df[col].nunique())
    print(df[col].value_counts().head(10))

# ==========================================================
# 7. OUTLIER ANALYSIS FOR EVERY NUMERICAL COLUMN
# ==========================================================

print("\n========== OUTLIER ANALYSIS ==========")

outlier_summary = []

for col in numerical_columns:

    data = df[col].dropna()

    if data.empty:
        continue

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data < lower_limit) | (data > upper_limit)
    ]

    outlier_summary.append({
        "Column": col,
        "Outlier Count": len(outliers),
        "Lower Limit": round(lower_limit, 2),
        "Upper Limit": round(upper_limit, 2)
    })

    print("\nColumn:", col)
    print("Potential Outliers:", len(outliers))
    print("Lower Limit:", round(lower_limit, 2))
    print("Upper Limit:", round(upper_limit, 2))

    plt.figure(figsize=(5, 6))
    sns.boxplot(y=data, color="skyblue")
    plt.title("Outlier Analysis: " + col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

print("\n========== OUTLIER SUMMARY TABLE ==========")

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
else:
    print("No numerical data available for outlier analysis.")

# ==========================================================
# 8. CORRELATION ANALYSIS
# ==========================================================

print("\n========== CORRELATION ANALYSIS ==========")

if len(numerical_columns) >= 2:

    correlation = df[numerical_columns].corr()

    print(correlation.round(2))

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

else:
    print("Not enough numerical columns for a heatmap.")

# ==========================================================
# 9. FINAL EDA SUMMARY
# ==========================================================

print("\n========== FINAL EDA SUMMARY ==========")

print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Total Missing Values:", df.isnull().sum().sum())
print("Total Duplicate Rows:", df.duplicated().sum())
print("Numerical Columns:", len(numerical_columns))
print("Categorical Columns:", len(categorical_columns))

print("\nEDA Completed Successfully!")

# ==========================================================
# 10. PREPARE DATA FOR PREDICTIVE ANALYSIS
# ==========================================================

print("\n========== DATA PREPARATION ==========")

if "Status" not in df.columns:
    raise ValueError(
        "The dataset does not contain a 'Status' column. "
        "Check the actual column names."
    )

# Keep only the two required classes
status_text = df["Status"].astype(str).str.strip().str.lower()

valid_rows = status_text.isin(["returned", "delivered"])

model_df = df.loc[valid_rows].copy()

if model_df.empty:
    raise ValueError(
        "No Returned or Delivered records were found. "
        "Check the values in the Status column."
    )

# Target: Returned = 1, Delivered = 0
y = (
    model_df["Status"].astype(str).str.strip().str.lower()
    == "returned"
).astype(int)

print("Records used for prediction:", len(model_df))
print("Returned Orders:", int(y.sum()))
print("Delivered Orders:", int((y == 0).sum()))

if y.nunique() < 2:
    raise ValueError(
        "Both Returned and Delivered classes are required."
    )

# Remove the target and columns that may leak the outcome
leakage_columns = [
    "Status",
    "Reason",
    "Rating",
    "Delivery Date",
    "OrderID",
    "CustomerID",
    "ReturnDate",
    "Return Date",
    "RefundAmount",
    "Refund Amount",
    "ReturnStatus",
    "Return Status"
]

# Match column names without case sensitivity
drop_columns = [
    col for col in model_df.columns
    if col.strip().lower() in {
        name.lower() for name in leakage_columns
    }
]

X = model_df.drop(columns=drop_columns, errors="ignore")

# Create features from the order date, if available
date_columns = [
    col for col in X.columns
    if col.strip().lower() == "orderdate"
]

for col in date_columns:

    order_date = pd.to_datetime(X[col], errors="coerce")

    X["OrderYear"] = order_date.dt.year
    X["OrderMonth"] = order_date.dt.month
    X["OrderDay"] = order_date.dt.day
    X["OrderDayOfWeek"] = order_date.dt.dayofweek

    X = X.drop(columns=[col])

# Remove columns that contain no usable values
X = X.dropna(axis=1, how="all")

# Remove columns that are constant
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_columns)

print("\nFeatures used:")
print(X.columns.tolist())

if X.shape[1] == 0:
    raise ValueError("No usable input features remain.")

# ==========================================================
# 11. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================================
# 12. PREPROCESSING
# ==========================================================

num_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ==========================================================
# 13. MODEL 1: LOGISTIC REGRESSION
# ==========================================================

print("\n========== MODEL 1: LOGISTIC REGRESSION ==========")

model1 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

accuracy1 = accuracy_score(y_test, pred1)

print("Accuracy:", round(accuracy1 * 100, 2), "%")

# ==========================================================
# 14. MODEL 2: K-NEAREST NEIGHBORS
# ==========================================================

print("\n========== MODEL 2: KNN ==========")

model2 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

accuracy2 = accuracy_score(y_test, pred2)

print("Accuracy:", round(accuracy2 * 100, 2), "%")

# ==========================================================
# 15. MODEL 3: DECISION TREE
# ==========================================================

print("\n========== MODEL 3: DECISION TREE ==========")

model3 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ))
])

model3.fit(X_train, y_train)

pred3 = model3.predict(X_test)

accuracy3 = accuracy_score(y_test, pred3)

print("Accuracy:", round(accuracy3 * 100, 2), "%")

# ==========================================================
# 16. MODEL 4: RANDOM FOREST
# ==========================================================

print("\n========== MODEL 4: RANDOM FOREST ==========")

model4 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])

model4.fit(X_train, y_train)

pred4 = model4.predict(X_test)

accuracy4 = accuracy_score(y_test, pred4)

print("Accuracy:", round(accuracy4 * 100, 2), "%")

# ==========================================================
# 17. MODEL 5: SUPPORT VECTOR MACHINE
# ==========================================================

print("\n========== MODEL 5: SVM ==========")

model5 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ))
])

model5.fit(X_train, y_train)

pred5 = model5.predict(X_test)

accuracy5 = accuracy_score(y_test, pred5)

print("Accuracy:", round(accuracy5 * 100, 2), "%")

# ==========================================================
# 18. MODEL COMPARISON
# ==========================================================

print("\n========== MODEL COMPARISON ==========")

predictions = {
    "Logistic Regression": pred1,
    "KNN": pred2,
    "Decision Tree": pred3,
    "Random Forest": pred4,
    "SVM": pred5
}

results = []

for name, predictions_model in predictions.items():

    results.append({
        "Model": name,
        "Accuracy (%)": accuracy_score(
            y_test, predictions_model
        ) * 100,
        "Precision (%)": precision_score(
            y_test, predictions_model, zero_division=0
        ) * 100,
        "Recall (%)": recall_score(
            y_test, predictions_model, zero_division=0
        ) * 100,
        "F1 Score (%)": f1_score(
            y_test, predictions_model, zero_division=0
        ) * 100
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print(results_df.to_string(
    index=False,
    formatters={
        "Accuracy (%)": "{:.2f}".format,
        "Precision (%)": "{:.2f}".format,
        "Recall (%)": "{:.2f}".format,
        "F1 Score (%)": "{:.2f}".format
    }
))

best_model = results_df.iloc[0]

print("\nBest model by accuracy:", best_model["Model"])
print("Best accuracy:", round(best_model["Accuracy (%)"], 2), "%")

# ==========================================================
# 19. CLASSIFICATION REPORTS
# ==========================================================

for name, predictions_model in predictions.items():

    print("\n==========", name.upper(), "REPORT ==========")

    print(classification_report(
        y_test,
        predictions_model,
        labels=[0, 1],
        target_names=["Delivered", "Returned"],
        zero_division=0
    ))

# ==========================================================
# 20. CONFUSION MATRIX FOR THE BEST MODEL
# ==========================================================

best_name = best_model["Model"]
best_predictions = predictions[best_name]

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Delivered", "Returned"],
    yticklabels=["Delivered", "Returned"]
)

plt.title("Confusion Matrix: " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n========== PREDICTIVE ANALYSIS COMPLETED ==========")
