from ucimlrepo import fetch_ucirepo 
import csv
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
# fetch dataset 
heart_disease = pd.read_csv('processed.cleveland.data', header=None, na_values='NaN')
heart_disease.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
target_column = "num"
print("Heart Disease Dataset:")
print(heart_disease.head())

heart_disease.hist(figsize=(10, 10))
plt.show()

# Convert categorical columns to 'category' dtype
heart_disease['cp'] = heart_disease['cp'].astype('category')
heart_disease['restecg'] = heart_disease['restecg'].astype('category')
heart_disease['slope'] = heart_disease['slope'].astype('category')
heart_disease['thal'] = heart_disease['thal'].astype('category')

# Convert categorical columns with binary representation
heart_disease['sex'] = heart_disease['sex'].map({0:0, 1:1})
heart_disease['fbs'] = heart_disease['fbs'].map({0:0, 1:1})
heart_disease['exang'] = heart_disease['exang'].map({0:0, 1:1})
heart_disease['num'] = heart_disease['num'].map({0:0, 1:1, 2:1, 3:1, 4:1})

categorical_category = [
    "cp", "restecg", "slope", "thal"
]
categorical_binary = [
    "sex", "fbs", "exang"
]

target_column = "num"

numerical_columns = [
    "trestbps", "chol", "thalach", "oldpeak", "ca"
]
categorical_columns = categorical_category + categorical_binary
feature_columns = heart_disease.columns
feature_columns.drop("num")

X = heart_disease[feature_columns]
y = heart_disease[target_column]

num_pipeline = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
]

cat_pipeline = [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=num_pipeline), numerical_columns),
        ("cat", Pipeline(steps=cat_pipeline), categorical_columns)
    ],
    remainder="drop"
)

full_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_proc = full_pipeline.fit_transform(X)

# Get feature names after preprocessing
cat_feature_names = list(full_pipeline.named_steps["preprocessor"]
                            .named_transformers_["cat"]
                            .named_steps["onehot"].get_feature_names_out(categorical_columns))
num_feature_names = numerical_columns
proc_feature_names = num_feature_names + cat_feature_names
X_proc_df = pd.DataFrame(X_proc, columns=proc_feature_names)
X_proc_df.hist(figsize=(10, 10))
plt.show()
print("Feature names after preprocessing:", proc_feature_names)

X = pd.get_dummies(heart_disease.drop(columns=[target_column]), drop_first=True)
y = heart_disease[target_column]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {'n_neighbors': range(1, 21)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
print(f"Best k: {best_k}")