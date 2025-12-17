##########################################
####### Framingham Heart Study ###########
##########################################
# importing needed libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


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
# Importing dataset
#######################################
df = pd.read_csv("data/framingham.csv")
print (" Data frame before cleaning")
print (df.head())

print (" Data frame before cleaning")
print(df.info())

# The target: 
# a prediction of whether a participant is estimated to develop Coronary Heart Disease (CHD) within 10 years.

df['TenYearCHD'].value_counts()
#Target distribution before cleaning
plt.pie(x=df['TenYearCHD'].value_counts(), labels=['low risk','high risk'], autopct='%1.1f%%')
plt.title("Target distribution before cleaning")
plt.show();

######################
####The features:####
######################
# Smoking => cigsPerDay

# how many null values in cigsPerDay?
df["cigsPerDay"].isnull().sum()

df["cigsPerDay"].value_counts()

# replace all nulls in (cigsPerDay) with 0, the most common value.

df['cigsPerDay'] = df['cigsPerDay'].isna().astype(int)
df['cigsPerDay'] = df['cigsPerDay'].fillna(0).astype(int)
df["cigsPerDay"].isnull().sum()

######################
##Total cholesterol ##
######################


df["totChol"].isnull().sum()



sns.histplot(df["totChol"])
plt.title ("Total Cholesterol")
plt.show(); 


# replace all 50 nulls with mean (thanks to the good distribution of the values)

df['totChol'] = df['totChol'].isna().astype(int)
totChol_mean = df['totChol'].mean() 
df['totChol'] = df['totChol'].fillna(totChol_mean).astype(int)
df["totChol"].isnull().sum()

# Obesity => BMI


df["BMI"].isnull().sum()
sns.histplot(df["BMI"])
plt.title ("Body mass index")
plt.show();  # not very good distribution,  rt skewed a bit, can replace nulls with median

# replace all nulls with median as the curve above is rt-skewed.


df['BMI'] = df['BMI'].isna().astype(int)
BMI_median = df['BMI'].median() 
df['BMI'] = df['BMI'].fillna(BMI_median).astype(int)
df["BMI"].isnull().sum()


#  Glucose level


df["glucose"].isnull().sum()


df[(df["glucose"].isnull()) & (df["TenYearCHD"] == 0)].shape[0]


# All 338 values will be removed as they will not worsen the imbalanced target.


df = df[~((df["glucose"].isnull()) & (df["TenYearCHD"] == 0))]
df["glucose"].isnull().sum()


sns.histplot(df["glucose"])
plt.title ("Blood Glucose level")
plt.show();

# > replace the remaining 50 nulls with median due to the rt-skewed data.


df['glucose'] = df['glucose'].isna().astype(int)
glucose_median = df["glucose"].median()
df['glucose'] = df['glucose'].fillna(glucose_median).astype(int)
df["glucose"].isnull().sum()


# #### Medications for hypertension


df['BPMeds'] = df['BPMeds'].fillna(0).astype(int) # 0, the most common value
df["BPMeds"].isnull().sum()


# #### Education


df[(df["education"].isnull()) & (df["TenYearCHD"] == 0)].shape[0]


df = df[~((df["education"].isnull()) & (df["TenYearCHD"] == 0))]# will remove from the 0s
df["education"].isnull().sum()


# Education plot
plt.figure(figsize=(10, 6))

# Count education levels
counts = df['education'].value_counts().sort_index()
colors = ['#E74C3C', '#3498DB', '#F1C40F', '#2ECC71']  # Red, Blue, Yellow, Green

# Plot
bars = plt.bar(range(1, 5), 
               [counts.get(i, 0) for i in range(1, 5)],
               color=colors,
               edgecolor='black')

# Labels
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4'], fontsize=12)
plt.xlabel('Education Level Code', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Education Level Distribution', fontweight='bold')

# Add legend
legend_labels = [
    '1: Less than High School',
    '2: High School Graduate/GED', 
    '3: Some College',
    '4: College Graduate'
]

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], 
                        edgecolor='black',
                        label=legend_labels[i]) 
                  for i in range(4)]

plt.legend(handles=legend_elements, 
           title='Code Explanation',
           loc='upper right',
           framealpha=0.9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show();



df['education'] = df['education'].fillna(1).astype(int) # 1 is the most common value
df["education"].isnull().sum()


df = df.dropna(subset=["heartRate"])


df.info()

########### correlation heatmap ##############
plt.figure (figsize = (12, 12))
sns.heatmap(df.corr(numeric_only= True), annot = True,  cmap = "coolwarm", vmin = -1, fmt =".1f")
plt.title ("Correlation map")
plt.show();




plt.pie(x=df['TenYearCHD'].value_counts(), labels=['low risk','high risk'], autopct='%1.1f%%')
plt.title("Target distribution after cleaning")
plt.show();

df.to_csv("data/framingham_cleaned.csv")
print("Cleaned data set")
print (df)

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


print(f"Best recall {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# best model
best_model = grid.best_estimator_
joblib.dump(best_model, "best_model.joblib")
pred = best_model.predict(X_test)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

##############################################
############  Confusion matrix  ##############
class_names = {
    0: 'No Heart Disease\n(Low Risk)',
    1: 'Heart Disease\n(High Risk)'
}

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(8, 6))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Reds',
            cbar=True,
            cbar_kws={'label': 'Number of Cases'},
            square=True)

plt.title('Heart Disease Prediction - Confusion Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Diagnosis', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Diagnosis', fontsize=12, fontweight='bold')

# Use descriptive labels
plt.xticks([0.5, 1.5], [class_names[0], class_names[1]], fontsize=11)
plt.yticks([0.5, 1.5], [class_names[0], class_names[1]], fontsize=11, rotation=0)

plt.tight_layout()
plt.show();



