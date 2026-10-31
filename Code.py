import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("FLIPKART_SHOW", "1") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ==========================================================
# SETTINGS AND HELPER FUNCTIONS
# ==========================================================

DATA_PATH = os.environ.get("FLIPKART_DATA", r"D:\243project\Flipkart Sales Dataset.xlsx")
OUT_DIR = os.environ.get("FLIPKART_OUT", r"D:\243project")
SHOW_PLOTS = os.environ.get("FLIPKART_SHOW", "1") == "1"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)


def finish(name):
    """Save the current figure as a PNG for the report, then show or close it."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=110, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def make_preprocessor(X):
    """Median-impute and scale numbers; mode-impute and one-hot encode text."""
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return pre, num, cat


def grouped_importance(pipe, cat_cols):
    """Sum one-hot importances back to the original column."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    out = {}
    for n, v in zip(names, imp):
        n = n.split("__", 1)[1]
        col = next((c for c in cat_cols if n.startswith(c + "_")), n)
        out[col] = out.get(col, 0) + v
    return pd.Series(out).sort_values(ascending=False)


# ##########################################################
#  PART 1 - DATA PREPROCESSING AND EDA
# ##########################################################

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_excel(DATA_PATH)

# Remember the original columns. Engineered columns added later
# (Delivery Days, Returned, Year...) are never given to the five models.
ORIGINAL_COLUMNS = df.columns.tolist()

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
# 2. MISSING AND NULL VALUES
# ==========================================================

print("\n========== MISSING / NULL VALUES ==========")

missing_values = df.isnull().sum()
missing_table = pd.DataFrame({
    "Missing": missing_values,
    "Percent (%)": (missing_values / len(df) * 100).round(2)
})

print(missing_table[missing_table["Missing"] > 0])
print("Total Missing Values:", int(missing_values.sum()))

# Empty text (blank strings) is also a kind of missing value
blank_counts = {}
for col in df.select_dtypes(exclude=["number", "datetime"]).columns:
    blank_counts[col] = int(df[col].dropna().astype(str).str.strip().eq("").sum())
print("\nBlank text cells per text column:",
      {k: v for k, v in blank_counts.items() if v > 0} or "none")

# Is the missing Reason explained by the order status?
if "Reason" in df.columns and "Status" in df.columns:
    print("\nMissing Reason by Status (Reason is only recorded for returns):")
    print(df.groupby("Status")["Reason"].apply(lambda s: int(s.isnull().sum())))

# ==========================================================
# 3. DUPLICATE RECORDS
# ==========================================================

print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate Rows:", df.duplicated().sum())
if "OrderID" in df.columns:
    print("Duplicate OrderIDs:", int(df["OrderID"].duplicated().sum()))

# ==========================================================
# 4. VALIDITY CHECKS
# ==========================================================

print("\n========== VALIDITY CHECKS ==========")

print("Delivery Date earlier than OrderDate:",
      int((df["Delivery Date"] < df["OrderDate"]).sum()))

for col in ["Unit Price", "Shipping Fee", "Order Quantity", "Sale Price"]:
    if col in df.columns:
        print(col, "- zero or negative values:", int((df[col] <= 0).sum()))

if "Rating" in df.columns:
    print("Rating outside 1-5:", int((~df["Rating"].between(1, 5)).sum()))
if "Customer Age" in df.columns:
    print("Customer Age outside 10-100:", int((~df["Customer Age"].between(10, 100)).sum()))

# ==========================================================
# 5. IDENTIFY COLUMN TYPES
# ==========================================================

numerical_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=["number", "datetime"]
).columns.tolist()

print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

# ==========================================================
# 6. DATA CLEANING AND FEATURE ENGINEERING
# ==========================================================

print("\n========== DATA CLEANING AND FEATURE ENGINEERING ==========")

# Remove stray spaces in text columns (missing values stay missing)
for col in categorical_columns:
    df[col] = df[col].str.strip()

# Features created from the dates and the status
df["Delivery Days"] = (df["Delivery Date"] - df["OrderDate"]).dt.days
df["Year"] = df["OrderDate"].dt.year
df["Month"] = df["OrderDate"].dt.month
df["DayOfWeek"] = df["OrderDate"].dt.dayofweek
df["Returned"] = (df["Status"].str.lower() == "returned").astype(int)

print("Created columns: Delivery Days, Year, Month, DayOfWeek, Returned")
print("Date range:", df["OrderDate"].min().date(), "to", df["OrderDate"].max().date())
print("\nDelivery Days summary:")
print(df["Delivery Days"].describe().round(2))

# ==========================================================
# 7. NUMERICAL SUMMARY
# ==========================================================

print("\n========== NUMERICAL SUMMARY ==========")

print(df[numerical_columns].describe().T.round(2).to_string())

# ==========================================================
# 8. CATEGORICAL SUMMARY
# ==========================================================

print("\n========== CATEGORICAL SUMMARY ==========")

for col in categorical_columns:
    print("\nColumn:", col)
    print("Unique Values:", df[col].nunique())
    print(df[col].value_counts().head(10))

# ==========================================================
# 9. OUTLIER ANALYSIS FOR EVERY NUMERICAL COLUMN
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
    finish("eda_boxplot_" + col.replace(" ", "_"))

print("\n========== OUTLIER SUMMARY TABLE ==========")

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
else:
    print("No numerical data available for outlier analysis.")

# ==========================================================
# 10. CORRELATION ANALYSIS AND HEATMAP
# ==========================================================

print("\n========== CORRELATION ANALYSIS ==========")

if len(numerical_columns) >= 2:

    correlation = df[numerical_columns].corr()

    print(correlation.round(2).to_string())

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
    finish("eda_correlation_heatmap")

else:
    print("Not enough numerical columns for a heatmap.")

# ==========================================================
# 11. FINAL EDA SUMMARY
# ==========================================================

print("\n========== FINAL EDA SUMMARY ==========")

print("Total Records:", df.shape[0])
print("Original Columns:", len(ORIGINAL_COLUMNS))
print("Total Missing Values:", int(df[ORIGINAL_COLUMNS].isnull().sum().sum()))
print("Total Duplicate Rows:", df.duplicated().sum())
print("Numerical Columns:", len(numerical_columns))
print("Categorical Columns:", len(categorical_columns))

print("\nEDA Completed Successfully!")



# ##########################################################
#  PART 2 - BUSINESS INSIGHTS
# ##########################################################

# ==========================================================
# 12. BUSINESS INSIGHTS
# ==========================================================
print("\n" + "=" * 60)
print("PART 2: BUSINESS INSIGHTS")
print("=" * 60)

by_year = df.groupby("Year").agg(Orders=("OrderID", "count"),
                                 Revenue=("Sale Price", "sum"))
print("\nOrders and revenue by year:\n", by_year)

by_cat = (df.groupby("Product Category")
            .agg(Orders=("OrderID", "count"), Revenue=("Sale Price", "sum"),
                 ReturnRate=("Returned", "mean"))
            .sort_values("Orders", ascending=False))
by_cat["ReturnRate"] = (by_cat["ReturnRate"] * 100).round(1)
print("\nBy product category:\n", by_cat)

by_loc = (df.groupby("Location")
            .agg(Orders=("OrderID", "count"), ReturnRate=("Returned", "mean"))
            .sort_values("Orders", ascending=False))
by_loc["ReturnRate"] = (by_loc["ReturnRate"] * 100).round(1)
print("\nTop 10 locations:\n", by_loc.head(10))

by_dt = (df.groupby("Delivery Type")
           .agg(Orders=("OrderID", "count"),
                AvgDeliveryDays=("Delivery Days", "mean"),
                ReturnRate=("Returned", "mean")))
by_dt["ReturnRate"] = (by_dt["ReturnRate"] * 100).round(1)
by_dt["AvgDeliveryDays"] = by_dt["AvgDeliveryDays"].round(2)
print("\nBy delivery type:\n", by_dt)

reasons = df.loc[df["Returned"] == 1, "Reason"].value_counts()
print("\nReturn reasons:\n", reasons)

reasons_share = (reasons / reasons.sum() * 100).round(1)
reasons_table = pd.DataFrame({"Orders": reasons, "Share (%)": reasons_share})
print("\nReturn reasons with share (%):\n", reasons_table)

growth = by_year["Orders"].pct_change() * 100
print("\nYear-on-year growth in orders (%):\n", growth.round(1))
top2_cat = by_cat["Orders"].head(2).sum() / len(df) * 100
top2_loc = by_loc["Orders"].head(2).sum() / len(df) * 100
print("\nTop 2 categories share of orders: %.1f%%" % top2_cat)
print("Top 2 locations share of orders: %.1f%%" % top2_loc)

df["Days Band"] = pd.cut(df["Delivery Days"], [0, 3, 6, 9, 12, 20],
                         labels=["<=3", "4-6", "7-9", "10-12", "13-20"])
band_ret = df.groupby("Days Band")["Returned"].mean() * 100
print("\nReturn rate (%) by delivery-time band:\n", band_ret.round(1))

rating_by_status = df.groupby("Status")["Rating"].mean().round(2)
print("\nAverage rating by status (rating is given after delivery, so it is\n"
      "reported here but NOT used as a predictor):\n", rating_by_status)

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
by_year["Orders"].plot(kind="bar", ax=ax[0, 0], color="steelblue")
ax[0, 0].set_title("Orders per Year (2020 is a spike)")
ax[0, 0].tick_params(axis="x", rotation=0)
by_cat["Orders"].plot(kind="barh", ax=ax[0, 1], color="seagreen")
ax[0, 1].set_title("Orders by Product Category")
by_loc["Orders"].head(10)[::-1].plot(kind="barh", ax=ax[0, 2], color="darkorange")
ax[0, 2].set_title("Top 10 Locations by Orders")
by_dt["ReturnRate"].plot(kind="bar", ax=ax[1, 0], color="indianred")
ax[1, 0].set_title("Return Rate (%) by Delivery Type")
ax[1, 0].tick_params(axis="x", rotation=15)
reasons[::-1].plot(kind="barh", ax=ax[1, 1], color="slategray")
ax[1, 1].set_title("Return Reasons")
band_ret.plot(kind="bar", ax=ax[1, 2], color="purple")
ax[1, 2].set_title("Return Rate (%) by Delivery Days")
ax[1, 2].tick_params(axis="x", rotation=0)
plt.suptitle("Business Insights", fontsize=16)
finish("business_insights")



# ##########################################################
#  PART 3 - FIVE CLASSIFICATION MODELS: RETURN PREDICTION
# ##########################################################

# ==========================================================
# 13. PREPARE DATA FOR PREDICTIVE ANALYSIS
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

# use only the original dataset columns, so the engineered columns
# (Delivery Days, Returned, Year...) can never leak into the models
model_df = df.loc[valid_rows, ORIGINAL_COLUMNS].copy()

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

# Remove features that showed no relationship with returns
noise_columns = [
    "Customer Gender", "Zone", "Product Category", "Order Quantity",
    "OrderYear", "OrderMonth", "OrderDay", "OrderDayOfWeek"
]

X = X.drop(columns=[c for c in noise_columns if c in X.columns])

print("\nFeatures used:")
print(X.columns.tolist())

if X.shape[1] == 0:
    raise ValueError("No usable input features remain.")

# ==========================================================
# 14. TRAIN-TEST SPLIT
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
# 15. PREPROCESSING
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
# 16. MODEL 1: LOGISTIC REGRESSION
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
# 17. MODEL 2: K-NEAREST NEIGHBORS
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
# 18. MODEL 3: DECISION TREE
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
# 19. MODEL 4: RANDOM FOREST
# ==========================================================

print("\n========== MODEL 4: RANDOM FOREST ==========")

model4 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

model4.fit(X_train, y_train)

pred4 = model4.predict(X_test)

accuracy4 = accuracy_score(y_test, pred4)

print("Accuracy:", round(accuracy4 * 100, 2), "%")

# ==========================================================
# 20. MODEL 5: SUPPORT VECTOR MACHINE
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
# 21. MODEL COMPARISON
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
# 22. CLASSIFICATION REPORTS
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
# 23. CONFUSION MATRIX FOR THE BEST MODEL
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
finish("model_confusion_matrix_best")


# ==========================================================
# 24. CONFUSION MATRICES FOR ALL FIVE MODELS
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, pred) in enumerate(predictions.items()):

    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    print("\n==========", name.upper(), "CONFUSION MATRIX ==========")
    print(cm)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Delivered", "Returned"],
        yticklabels=["Delivered", "Returned"],
        linewidths=0.5,
        ax=axes[i]
    )

    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted Status")
    axes[i].set_ylabel("Actual Status")

axes[5].axis("off")

plt.suptitle("Confusion Matrices of Five Models", fontsize=16)
plt.tight_layout()
finish("model_confusion_matrices_all")


# ==========================================================
# 25. ACCURACY COMPARISON
# ==========================================================

plt.figure(figsize=(10, 6))

accuracy_data = results_df.sort_values(
    "Accuracy (%)", ascending=False
)

ax = sns.barplot(
    data=accuracy_data,
    x="Model",
    y="Accuracy (%)"
)

plt.title("Accuracy Comparison of Five Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.tight_layout()
finish("model_accuracy_comparison")


# ==========================================================
# 26. PRECISION, RECALL AND F1-SCORE COMPARISON
# ==========================================================

metric_columns = [
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)"
]

results_df.set_index("Model")[metric_columns].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 100)
)

plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score (%)")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Evaluation Metric")
plt.tight_layout()
finish("model_precision_recall_f1")


# ==========================================================
# 27. ACTUAL VS PREDICTED ORDER DISTRIBUTION
# ==========================================================

distribution_data = {
    "Actual": [
        int((y_test == 0).sum()),
        int((y_test == 1).sum())
    ]
}

for name, pred in predictions.items():

    distribution_data[name] = [
        int((pred == 0).sum()),
        int((pred == 1).sum())
    ]

distribution_df = pd.DataFrame(
    distribution_data,
    index=["Delivered", "Returned"]
)

distribution_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Actual vs Predicted Order Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.legend(title="Actual / Model")
plt.tight_layout()
finish("model_actual_vs_predicted_distribution")


# ==========================================================
# 28. RETURNED-ORDER DETECTION PERFORMANCE
# ==========================================================

returned_metrics = []

for name, pred in predictions.items():

    returned_metrics.append({
        "Model": name,
        "Precision": precision_score(
            y_test, pred, pos_label=1, zero_division=0
        ) * 100,

        "Recall": recall_score(
            y_test, pred, pos_label=1, zero_division=0
        ) * 100,

        "F1 Score": f1_score(
            y_test, pred, pos_label=1, zero_division=0
        ) * 100
    })

returned_df = pd.DataFrame(returned_metrics)

returned_df.set_index("Model").plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 100)
)

plt.title("Returned-Order Detection Performance")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score (%)")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Metric")
plt.tight_layout()
finish("model_returned_detection")


# ==========================================================
# 29. FINAL MODEL SUMMARY
# ==========================================================

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results_df.sort_values(
        "Accuracy (%)", ascending=False
    ).round(2).to_string(index=False)
)

best_accuracy_name = results_df.loc[
    results_df["Accuracy (%)"].idxmax(), "Model"
]

best_f1_name = results_df.loc[
    results_df["F1 Score (%)"].idxmax(), "Model"
]

print("\nBest Model by Accuracy:", best_accuracy_name)
print("Best Model by Returned-Class F1 Score:", best_f1_name)

#print("\nAll predictive analysis visualizations completed!")


# ==========================================================
# 30. CALIBRATED SVM FOR RETURN PROBABILITY
# ==========================================================

print("\n========== CALIBRATING SVM PROBABILITIES ==========")

calibrated_svm = CalibratedClassifierCV(
    estimator=model5,
    method="sigmoid",
    cv=3
)

calibrated_svm.fit(X_train, y_train)

# All five models, with calibrated SVM for probabilities
probability_models = {
    "Logistic Regression": model1,
    "KNN": model2,
    "Decision Tree": model3,
    "Random Forest": model4,
    "SVM": calibrated_svm
}

# ==========================================================
# 31. ACTUAL VS PREDICTED ORDER STATUS
# ==========================================================

print("\n========== ACTUAL VS PREDICTED STATUS ==========")

comparison_df = pd.DataFrame({
    "Actual Status": y_test.map({
        0: "Delivered",
        1: "Returned"
    })
})

for name, model in probability_models.items():

    predicted = model.predict(X_test)

    comparison_df[name] = pd.Series(
        predicted,
        index=X_test.index
    ).map({
        0: "Delivered",
        1: "Returned"
    })

print(comparison_df.head(20).to_string())

# Plot actual and predicted status for the first 20 test orders
plot_df = pd.DataFrame({
    "Actual": y_test.map({0: 0, 1: 1})
})

for name, model in probability_models.items():
    plot_df[name] = model.predict(X_test)

plot_df = plot_df.head(20)

plt.figure(figsize=(14, 6))

plt.plot(
    range(1, len(plot_df) + 1),
    plot_df["Actual"],
    marker="o",
    linewidth=3,
    label="Actual Status"
)

for name in probability_models:
    plt.plot(
        range(1, len(plot_df) + 1),
        plot_df[name],
        marker="x",
        linestyle="--",
        label=name
    )

plt.yticks([0, 1], ["Delivered", "Returned"])
plt.xlabel("Test Order Number")
plt.ylabel("Order Status")
plt.title("Actual vs Predicted Status for 20 Test Orders")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
finish("model_actual_vs_predicted_20_orders")

# ==========================================================
# 32. ROC CURVES FOR ALL FIVE MODELS
# ==========================================================

plt.figure(figsize=(10, 7))

for name, model in probability_models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities,
        pos_label=1
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Five Models")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
finish("model_roc_curves")

# ==========================================================
# 33. RETURN PROBABILITY COMPARISON
# ==========================================================

probability_results = {}

for name, model in probability_models.items():
    probability_results[name] = (
        model.predict_proba(X_test)[:, 1] * 100
    )

probability_df = pd.DataFrame(
    probability_results,
    index=X_test.index
)

# Select the best model by returned-class F1 score
best_f1_name = results_df.loc[
    results_df["F1 Score (%)"].idxmax(),
    "Model"
]

print("\nBest model by returned-class F1 score:", best_f1_name)

# Show the first 20 test orders
sample_probabilities = probability_df.head(20)

plt.figure(figsize=(13, 7))

for name in probability_models:
    plt.plot(
        range(1, len(sample_probabilities) + 1),
        sample_probabilities[name],
        marker="o",
        linewidth=1.5,
        label=name
    )

plt.axhline(
    50,
    linestyle="--",
    linewidth=1,
    label="50% Reference"
)

plt.xlabel("Test Order Number")
plt.ylabel("Estimated Return Probability (%)")
plt.title("Return Probability Comparison for 20 Orders")
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
finish("model_return_probability_20_orders")

# ==========================================================
# 34. INDIVIDUAL ORDER PREDICTIONS
# ==========================================================

individual_predictions = pd.DataFrame(index=X_test.index)

# Include order ID if the original dataset contains it
order_id_col = next(
    (
        col for col in model_df.columns
        if col.strip().lower() == "orderid"
    ),
    None
)

if order_id_col is not None:
    individual_predictions["Order ID"] = (
        model_df.loc[X_test.index, order_id_col]
    )
else:
    individual_predictions["Order Index"] = X_test.index

individual_predictions["Actual Status"] = y_test.map({
    0: "Delivered",
    1: "Returned"
})

for name, model in probability_models.items():

    predicted = model.predict(X_test)
    return_probability = model.predict_proba(X_test)[:, 1] * 100

    individual_predictions[name + " Prediction"] = pd.Series(
        predicted,
        index=X_test.index
    ).map({
        0: "Delivered",
        1: "Returned"
    })

    individual_predictions[name + " Return Probability (%)"] = (
        return_probability.round(2)
    )

print("\n========== INDIVIDUAL ORDER PREDICTIONS ==========")
print(individual_predictions.head(20).to_string(index=False))

# ==========================================================
# 35. VISUALIZE INDIVIDUAL RETURN PREDICTIONS
# ==========================================================

best_model_probability = probability_df[best_f1_name].head(20)
actual_sample = y_test.loc[best_model_probability.index]

colors = [
    "tab:red" if actual == 1 else "tab:blue"
    for actual in actual_sample
]

plt.figure(figsize=(12, 6))

bars = plt.bar(
    range(1, len(best_model_probability) + 1),
    best_model_probability,
    color=colors
)

plt.axhline(
    50,
    linestyle="--",
    label="50% Reference"
)

plt.xticks(range(1, len(best_model_probability) + 1))
plt.xlabel("Test Order Number")
plt.ylabel("Estimated Return Probability (%)")
plt.title(
    "Individual Return Predictions - " + best_f1_name
)
plt.ylim(0, 100)
plt.legend(["50% Reference", "Actual Returned", "Actual Delivered"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
finish("model_individual_return_predictions")

# ==========================================================
# 36. EXPORT PREDICTIONS TO EXCEL
# ==========================================================

output_path = os.path.join(OUT_DIR, "Flipkart_Return_Predictions.xlsx")

individual_predictions.to_excel(
    output_path,
    index=False
)

print("\nPrediction file saved successfully:")
print(output_path)

print("\nAll additional predictive visualizations completed!")



# ##########################################################
#  PART 4 - EFFECT OF DELIVERY TIME ON RETURN PREDICTION
# ##########################################################

# ==========================================================
# 37. DOES DELIVERY TIME HELP TO PREDICT RETURNS?
# ==========================================================
print("\n" + "=" * 60)
print("PART 4: DOES DELIVERY TIME HELP TO PREDICT RETURNS?")
print("=" * 60)

cls_df = df[df["Status"].str.strip().str.lower().isin(["returned", "delivered"])]
yc = cls_df["Returned"]
base_feats = ["Customer Age", "Location", "Delivery Type", "SubCategory", "Product",
              "Unit Price", "Shipping Fee"]
feature_sets = {
    "Without delivery days": base_feats,
    "With delivery days": base_feats + ["Delivery Days"],
}

tr_idx, te_idx = train_test_split(
    cls_df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=yc)
yc_tr, yc_te = yc.loc[tr_idx], yc.loc[te_idx]

rows, roc_store = [], {}
for fs_name, feats in feature_sets.items():
    Xc_tr, Xc_te = cls_df.loc[tr_idx, feats], cls_df.loc[te_idx, feats]
    pre_c, _, _ = make_preprocessor(Xc_tr)
    models_c = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    }
    for name, m in models_c.items():
        pipe = Pipeline([("pre", pre_c), ("model", m)]).fit(Xc_tr, yc_tr)
        pred = pipe.predict(Xc_te)
        proba = (pipe.predict_proba(Xc_te)[:, 1]
                 if hasattr(m, "predict_proba") else np.zeros(len(pred)))
        auc_value = roc_auc_score(yc_te, proba) if name != "Baseline (majority)" else 0.5
        rows.append({
            "Features": fs_name, "Model": name,
            "Accuracy %": 100 * accuracy_score(yc_te, pred),
            "Precision %": 100 * precision_score(yc_te, pred, zero_division=0),
            "Recall %": 100 * recall_score(yc_te, pred, zero_division=0),
            "F1 %": 100 * f1_score(yc_te, pred, zero_division=0),
            "ROC AUC": auc_value,
        })
        roc_store[(fs_name, name)] = (yc_te, proba)

res_d = pd.DataFrame(rows)
print(res_d.round(3).to_string(index=False))

print("\nROC AUC comparison (0.5 = random guessing):")
auc_cmp = res_d[res_d["Model"] != "Baseline (majority)"].pivot(
    index="Model", columns="Features", values="ROC AUC")
auc_cmp["Gain from delivery days"] = (auc_cmp["With delivery days"]
                                      - auc_cmp["Without delivery days"])
print(auc_cmp.round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
auc_cmp[["Without delivery days", "With delivery days"]].plot(kind="bar", ax=ax[0])
ax[0].axhline(0.5, color="red", linestyle="--", label="Random guess")
ax[0].set_ylim(0.4, 0.7)
ax[0].set_title("ROC AUC: Effect of Adding Delivery Days")
ax[0].tick_params(axis="x", rotation=15)
ax[0].legend()
for (fs_name, name), (yt, pr) in roc_store.items():
    if fs_name == "With delivery days" and name != "Baseline (majority)":
        fpr, tpr, _ = roc_curve(yt, pr)
        ax[1].plot(fpr, tpr, label="%s (AUC %.3f)" % (name, roc_auc_score(yt, pr)))
ax[1].plot([0, 1], [0, 1], "k--", label="Random")
ax[1].set_xlabel("False positive rate")
ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC Curves (with delivery days)")
ax[1].legend()
finish("delivery_days_effect")



# ##########################################################
#  PART 5 - DELIVERY-TIME REGRESSION
# ##########################################################

# ==========================================================
# 38. DELIVERY-TIME REGRESSION
# ==========================================================
print("\n" + "=" * 60)
print("PART 5: REGRESSION - PREDICTING DELIVERY DAYS")
print("=" * 60)

feat_b = ["Customer Age", "Customer Gender", "Location", "Zone", "Delivery Type",
          "Product Category", "SubCategory", "Product", "Unit Price",
          "Shipping Fee", "Order Quantity", "Year", "Month", "DayOfWeek"]
Xb, yb = df[feat_b], df["Delivery Days"].astype(float)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.2, random_state=RANDOM_STATE)

pre_b, _, cat_b = make_preprocessor(Xb_tr)
models_b = {
    "Baseline (mean)": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1),
}
fit_b, pred_b, rows = {}, {}, []
for name, m in models_b.items():
    pipe = Pipeline([("pre", pre_b), ("model", m)]).fit(Xb_tr, yb_tr)
    p = pipe.predict(Xb_te)
    fit_b[name], pred_b[name] = pipe, p
    rows.append({"Model": name, "R2": r2_score(yb_te, p),
                 "MAE (days)": mean_absolute_error(yb_te, p),
                 "RMSE (days)": np.sqrt(mean_squared_error(yb_te, p))})
res_b = pd.DataFrame(rows).sort_values("R2", ascending=False)
print(res_b.round(4).to_string(index=False))
best_b = res_b.iloc[0]["Model"]
print("\nBest model:", best_b)

# Which inputs matter?
imp_b = grouped_importance(fit_b["Random Forest"], cat_b)
print("\nFeature importance (Random Forest, grouped by original column):")
print(imp_b.head(8).round(4).to_string())

# Error by delivery type
err = pd.DataFrame({"type": Xb_te["Delivery Type"],
                    "abs_err": np.abs(yb_te - pred_b[best_b])})
print("\nMean absolute error (days) by delivery type:")
print(err.groupby("type")["abs_err"].mean().round(2).to_string())

fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
sns.barplot(data=res_b, x="Model", y="R2", ax=ax[0])
ax[0].set_title("Delivery Days: R2 by Model")
ax[0].tick_params(axis="x", rotation=25)
for c in ax[0].containers:
    ax[0].bar_label(c, fmt="%.3f")
idx = np.random.RandomState(0).choice(len(yb_te), 3000, replace=False)
ax[1].scatter(yb_te.values[idx], pred_b[best_b][idx], alpha=0.3, s=10)
ax[1].plot([0, 20], [0, 20], "r--")
ax[1].set_title("Actual vs Predicted Days: " + best_b)
ax[1].set_xlabel("Actual days")
ax[1].set_ylabel("Predicted days")
imp_b.head(8)[::-1].plot(kind="barh", ax=ax[2], color="teal")
ax[2].set_title("What Drives Delivery Time")
finish("delivery_time_regression")



# ##########################################################
#  PART 6 - MONTHLY ORDER FORECAST
# ##########################################################

# ==========================================================
# 39. MONTHLY ORDER FORECAST AND ANOMALY DETECTION
# ==========================================================
print("\n" + "=" * 60)
print("PART 6: MONTHLY ORDER FORECAST AND ANOMALY DETECTION")
print("=" * 60)

monthly = (df.groupby(pd.Grouper(key="OrderDate", freq="MS"))
             .agg(Orders=("OrderID", "count"), Revenue=("Sale Price", "sum"))
             .reset_index())
monthly["t"] = np.arange(len(monthly))
monthly["Month"] = monthly["OrderDate"].dt.month
monthly["Year"] = monthly["OrderDate"].dt.year

# Train on 2015-2019, then predict 2020 as unseen data.
train_m = monthly[monthly["Year"] <= 2019]
test_m = monthly[monthly["Year"] == 2020].copy()


def month_design(d):
    return pd.concat([d[["t"]].reset_index(drop=True),
                      pd.get_dummies(d["Month"].astype(
                          pd.CategoricalDtype(range(1, 13))), prefix="m").astype(int)
                      .reset_index(drop=True)], axis=1)


Xm_tr, Xm_te = month_design(train_m), month_design(test_m)
ym_tr = train_m["Orders"].values

fc = {"Baseline (mean of 2015-19)": np.full(len(test_m), ym_tr.mean())}
last_year = train_m[train_m["Year"] == 2019].set_index("Month")["Orders"]
fc["Seasonal naive (same month 2019)"] = test_m["Month"].map(last_year).values
fc["Linear (trend + month)"] = LinearRegression().fit(Xm_tr, ym_tr).predict(Xm_te)
fc["Random Forest"] = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE
).fit(Xm_tr, ym_tr).predict(Xm_te)

normal = test_m["Month"].values >= 6   # Jun-Dec 2020 behave normally
rows = []
for name, p in fc.items():
    rows.append({
        "Model": name,
        "MAE Jun-Dec 2020": mean_absolute_error(test_m["Orders"][normal], p[normal]),
        "MAPE % Jun-Dec 2020": 100 * np.mean(
            np.abs(test_m["Orders"].values[normal] - p[normal])
            / test_m["Orders"].values[normal]),
    })
res_c = pd.DataFrame(rows).sort_values("MAE Jun-Dec 2020")
print("Forecast accuracy on the 'normal' months Jun-Dec 2020:")
print(res_c.round(2).to_string(index=False))
best_c = res_c.iloc[0]["Model"]
print("\nBest forecaster:", best_c)

test_m["Forecast"] = fc[best_c]
test_m["Actual / Forecast"] = test_m["Orders"] / test_m["Forecast"]
print("\n2020 actual vs forecast:")
print(test_m[["OrderDate", "Orders", "Forecast", "Actual / Forecast"]]
      .round(2).to_string(index=False))
anom = test_m[test_m["Actual / Forecast"] > 1.5]
print("\nANOMALY: months where orders were more than 1.5x the forecast:",
      [d.strftime("%b %Y") for d in anom["OrderDate"]])
print("Average uplift in those months: %.2fx" % anom["Actual / Forecast"].mean())
print("Seasonality note: after excluding 2020, monthly averages are flat, so the\n"
      "apparent Jan-May 'peak' comes from the 2020 spike, not from a yearly cycle.")

plt.figure(figsize=(14, 5.5))
plt.plot(monthly["OrderDate"], monthly["Orders"], label="Actual", color="steelblue")
plt.plot(test_m["OrderDate"], test_m["Forecast"], "--", color="red",
         label="Forecast (" + best_c + ")")
if len(anom):
    plt.axvspan(anom["OrderDate"].min(),
                anom["OrderDate"].max() + pd.offsets.MonthEnd(0),
                color="orange", alpha=0.25, label="Anomalous surge")
plt.axvline(pd.Timestamp("2020-01-01"), color="gray", linestyle=":")
plt.title("Monthly Orders: Forecast vs Actual")
plt.ylabel("Orders")
plt.legend()
finish("monthly_forecast")



# ##########################################################
#  PART 7 - SALE PRICE TEST AND LIMITATIONS
# ##########################################################

# ==========================================================
# 40. SALE PRICE REGRESSION TEST AND DATASET LIMITATIONS
# ==========================================================
print("\n" + "=" * 60)
print("PART 7: SALE PRICE TEST AND LIMITATIONS OF THE DATASET")
print("=" * 60)

num_check = ["Customer Age", "Unit Price", "Shipping Fee", "Order Quantity", "Rating",
             "Delivery Days"]
corr_sp = df[num_check].corrwith(df["Sale Price"]).round(4)
print("Correlation of Sale Price with other numeric columns:\n", corr_sp.to_string())
cat_means = df.groupby("Product Category")["Sale Price"].mean().round(1)
print("\nAverage Sale Price by category (almost identical):\n", cat_means.to_string())
gap = (df["Unit Price"] * df["Order Quantity"] + df["Shipping Fee"] - df["Sale Price"])
print("\nUnit Price x Quantity + Shipping minus Sale Price: mean %.1f, std %.1f"
      % (gap.mean(), gap.std()))
# ----------------------------------------------------------
# Can Sale Price be predicted? Six regression models
# ----------------------------------------------------------
Xs, ys = df[feat_b], df["Sale Price"].astype(float)

num_feats = Xs.select_dtypes(include="number").columns
near_copy = Xs[num_feats].corrwith(ys).abs()
near_copy = near_copy[near_copy > 0.95].index.tolist()
print("\nColumns that are near-copies of Sale Price (correlation > 0.95):",
      near_copy if near_copy else "none")

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    Xs, ys, test_size=0.2, random_state=RANDOM_STATE)
pre_s, _, _ = make_preprocessor(Xs_tr)

models_s = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=10000),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1),
}
rows = []
for name, m in models_s.items():
    pipe = Pipeline([("pre", pre_s), ("model", m)]).fit(Xs_tr, ys_tr)
    p = pipe.predict(Xs_te)
    rows.append({"Model": name, "R2": r2_score(ys_te, p),
                 "MAE": mean_absolute_error(ys_te, p),
                 "RMSE": np.sqrt(mean_squared_error(ys_te, p))})
res_sp = pd.DataFrame(rows).sort_values("R2", ascending=False)
print("\nSale Price regression results (R2 near 0 = no better than the average):")
print(res_sp.round(4).to_string(index=False))

plt.figure(figsize=(8, 4.5))
ax_sp = sns.barplot(data=res_sp, x="Model", y="R2")
plt.axhline(0, color="black", linewidth=1)
plt.title("Predicting Sale Price: R2 of Six Models")
plt.xticks(rotation=20, ha="right")
for c in ax_sp.containers:
    ax_sp.bar_label(c, fmt="%.4f", padding=3)
finish("sale_price_regression")
