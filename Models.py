
import pandas as pd
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
from imblearn.pipeline import make_pipeline as make_imb_pipeline

##########################################################################################

# #### Importing data and train test split


df = pd.read_csv("data/framingham_PROPERLY_cleaned.csv")

X = df.drop(["TenYearCHD"], axis = 1)
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 11)

##########################################################################################
# ## 1. Logistic regression 
##########################################################################################
# ### without undersampling or over sampling
##########################################################################################


pipeline = Pipeline(steps = [
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


print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_


pred = best_model.predict(X_test)
print ("Log. reg  metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# ### With random under sampler
##########################################################################################


pipeline = Pipeline(steps = [
    ('undersample', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
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


print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_

pred = best_model.predict(X_test)
print ("Log. reg + random under sampler metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# ### SMOTE
##########################################################################################


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


print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_

pred = best_model.predict(X_test)
print ("Log. reg + SMOTE metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# ## 2. Random forest classifier
##########################################################################################

pipeline = Pipeline(steps = [
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])


param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 20, None],
    'model__class_weight': [{0:1, 1:3}, {0:1, 1:5}, 'balanced', None],
    
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

print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_

pred = best_model.predict(X_test)
print ("Random forest classifier metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# ##  3. Gradient Boosting classifier
##########################################################################################

pipeline = Pipeline(steps = [
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),  # Note: Scaling not strictly needed for tree-based models
    ('model', GradientBoostingClassifier(random_state=42))
])

# Updated parameter grid for Gradient Boosting
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    
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

print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_

pred = best_model.predict(X_test)
print ("Gradient Boosting classifier metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# ## 4. KNN
##########################################################################################

pipeline = Pipeline(steps = [
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()), 
    ('model', KNeighborsClassifier())
])


param_grid = {
    'model__n_neighbors': [3, 5, 7, 9, 11],
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2]
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

print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")


best_model = grid.best_estimator_

pred = best_model.predict(X_test)
print ("KNN metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


##########################################################################################
# ## 5. XGBoost classifier
##########################################################################################
pipeline = Pipeline(steps = [
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('model', XGBClassifier(random_state=42, 
                           eval_metric='logloss',
                           use_label_encoder=False))
])


param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.2, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__scale_pos_weight': [1, 3, 5]
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

print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_

pred = best_model.predict(X_test)
print ("XGBoost classifier  metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# ## 6. k-means (unsupervised)
##########################################################################################


kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)


X_train_with_risk = X_train.copy()
X_test_with_risk = X_test.copy()

X_train_with_risk['risk_cluster'] = kmeans.fit_predict(X_train)
X_test_with_risk['risk_cluster'] = kmeans.predict(X_test)


pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

param_grid = {
    'model__C': [0.01, 0.05, 1, 10],
    'model__class_weight': [{0:1, 1:3}, {0:1, 1:5}, 'balanced'],
    'model__solver': ['liblinear', 'lbfgs'],
    'model__max_iter': [100, 1000]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1,
)

# Train on data with risk clusters
grid.fit(X_train_with_risk, y_train)

pred = best_model.predict(X_test)
print ("k-means  metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##########################################################################################
# 7. SVC
##########################################################################################

from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler  # Important for SVC!
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),  
    ('model', SVC(random_state=42, probability=True)) 
])

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'model__kernel': ['linear', 'rbf', 'poly'],  # Kernel types
    'model__gamma': ['scale', 'auto'],
    'model__class_weight': ['balanced', {0:1, 1:3}]
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

print(f"Best recall: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")

# Best model
best_model = grid.best_estimator_
pred = best_model.predict(X_test)
print ("SVC metrics")
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
##########################################################################################

