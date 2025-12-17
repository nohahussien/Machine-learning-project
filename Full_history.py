##########################################
####### Framingham Heart Study ###########
##########################################
# importing needed libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. SUPERVISED MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 2. UNSUPERVISED
from sklearn.cluster import KMeans

# 3. SAMPLING TECHNIQUES
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek

# 4. PIPELINE
from imblearn.pipeline import Pipeline

#######################################
# Importing dataset - ALWAYS LOAD FROM ORIGINAL!
#######################################
print("="*60)
print("LOADING ORIGINAL DATA FROM: data/framingham.csv")
print("="*60)

# Load from ORIGINAL file, not the cleaned one!
df = pd.read_csv("data/framingham.csv")

print("First 5 rows of ORIGINAL data:")
print(df.head())

print("\nOriginal data info:")
print(df.info())

# Check if data is already corrupted
print("\n" + "="*60)
print("CHECKING DATA INTEGRITY")
print("="*60)

# Sample original values before any processing
sample_before = df[['cigsPerDay', 'totChol', 'BMI', 'glucose']].head().copy()
print("Sample values BEFORE cleaning (should show proper numbers):")
print(sample_before)

######################
#### DATA CLEANING ####
######################

# 1. cigsPerDay - fill nulls with 0
print("\n1. Cleaning cigsPerDay...")
print(f"   Null count: {df['cigsPerDay'].isnull().sum()}")
df['cigsPerDay'] = df['cigsPerDay'].fillna(0)
print(f"   After filling: {df['cigsPerDay'].isnull().sum()}")

# 2. totChol - fill nulls with mean
print("\n2. Cleaning totChol...")
print(f"   Null count: {df['totChol'].isnull().sum()}")
totChol_mean = df['totChol'].mean()
print(f"   Mean value: {totChol_mean:.2f}")
df['totChol'] = df['totChol'].fillna(totChol_mean)
print(f"   After filling: {df['totChol'].isnull().sum()}")

# 3. BMI - fill nulls with median
print("\n3. Cleaning BMI...")
print(f"   Null count: {df['BMI'].isnull().sum()}")
BMI_median = df['BMI'].median()
print(f"   Median value: {BMI_median:.2f}")
df['BMI'] = df['BMI'].fillna(BMI_median)
print(f"   After filling: {df['BMI'].isnull().sum()}")

# 4. Glucose - remove rows where glucose is null AND TenYearCHD == 0
print("\n4. Cleaning glucose...")
print(f"   Null count: {df['glucose'].isnull().sum()}")
rows_to_remove = df[(df["glucose"].isnull()) & (df["TenYearCHD"] == 0)].shape[0]
print(f"   Removing {rows_to_remove} rows (null glucose & TenYearCHD=0)")
df = df[~((df["glucose"].isnull()) & (df["TenYearCHD"] == 0))]

# Fill remaining nulls with median
glucose_median = df["glucose"].median()
print(f"   Glucose median: {glucose_median:.2f}")
df['glucose'] = df['glucose'].fillna(glucose_median)
print(f"   After filling: {df['glucose'].isnull().sum()}")

# 5. BPMeds - fill nulls with 0
print("\n5. Cleaning BPMeds...")
print(f"   Null count: {df['BPMeds'].isnull().sum()}")
df['BPMeds'] = df['BPMeds'].fillna(0)
print(f"   After filling: {df['BPMeds'].isnull().sum()}")

# 6. Education - remove rows where education is null AND TenYearCHD == 0
print("\n6. Cleaning education...")
print(f"   Null count: {df['education'].isnull().sum()}")
rows_to_remove = df[(df["education"].isnull()) & (df["TenYearCHD"] == 0)].shape[0]
print(f"   Removing {rows_to_remove} rows (null education & TenYearCHD=0)")
df = df[~((df["education"].isnull()) & (df["TenYearCHD"] == 0))]

# Fill remaining nulls with 1
df['education'] = df['education'].fillna(1)
print(f"   After filling: {df['education'].isnull().sum()}")

# 7. heartRate - drop any remaining nulls
print("\n7. Cleaning heartRate...")
print(f"   Null count: {df['heartRate'].isnull().sum()}")
df = df.dropna(subset=["heartRate"])
print(f"   After dropping: {df['heartRate'].isnull().sum()}")

# Check final results
print("\n" + "="*60)
print("CLEANING COMPLETE - VERIFICATION")
print("="*60)

print("\nSample values AFTER cleaning (should show proper numbers):")
sample_after = df[['cigsPerDay', 'totChol', 'BMI', 'glucose']].head()
print(sample_after)

print("\nChecking if values are proper (not all 0s):")
for col in ['cigsPerDay', 'totChol', 'BMI', 'glucose']:
    unique_vals = sorted(df[col].unique())[:10]
    print(f"{col:12s}: {len(unique_vals)} unique values, sample: {unique_vals}")

print("\nSummary statistics:")
print(df[['cigsPerDay', 'totChol', 'BMI', 'glucose']].describe())

print("\n" + "="*60)
print("FINAL DATASET INFO")
print("="*60)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nRemaining nulls:")
print(df.isnull().sum())

# Save to NEW file (not overwriting original)
df.to_csv("data/framingham_PROPERLY_cleaned.csv", index=False)
print("\nSaved properly cleaned data to: data/framingham_PROPERLY_cleaned.csv")

##########################################################
###############   Model: Logistic regression  ############
###################  oversampling: SMOTE    ##############
##########################################################

X = df.drop(["TenYearCHD"], axis = 1)
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 11)

pipeline = Pipeline(steps = [
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

param_grid = {
    'model__C': [0.01, 0.05, 1, 10],
    'model__class_weight': [{0:1, 1:3}, {0:1, 1:5}, 'balanced'],
    'model__solver': ['liblinear', 'lbfgs'],
    'model__max_iter': [100, 1000, 10000]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

print(f"\nBest recall: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_
joblib.dump(best_model, "best_model.joblib")
pred = best_model.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, pred):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print(f"\nClassification Report:")
print(classification_report(y_test, pred))