# Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Data Exploration

# Load dataset
hd_df = pd.read_csv('processed.cleveland.data.csv')

# Examine structure
print("Dataset Shape: \n", hd_df.shape)
print("Dataset Columns: \n", hd_df.columns.tolist())
print("First 5 Rows: \n", hd_df.head())

# Target column reprocessing
hd_df['num'] = hd_df['num'].map({0:0, 1:1, 2:1, 3:1, 4:1})

missing_values = hd_df.isnull().sum()
print("Missing Values in Each Column: \n", missing_values)

num_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
print("Summary Statistics of Numeric Features: \n", hd_df[num_features].describe())

cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in cat_features:
    print(f"Value Counts for {col}: \n", hd_df[col].value_counts(dropna=False))

# Data Preprocessing

# Handle Outliers
for col in cat_features:
    hd_df[col] = hd_df[col].astype('category')

hd_encoded = pd.get_dummies(hd_df, columns=cat_features, drop_first=True)
print("Encoded Shape: \n", hd_encoded.shape)


for col in num_features:
    Q1 = hd_df[col].quantile(0.25)
    Q3 = hd_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = hd_df[(hd_df[col] < Q1 - 1.5 * IQR) | (hd_df[col] > Q3 + 1.5 * IQR)]
    if not outliers.empty:
        print(f"Outliers in {col}: \n", outliers[col])
        print(f"Number of outliers in {col}: \n", outliers[col].shape[0])
        plt.figure()
        sns.boxplot(x=hd_df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

Q1 = hd_df['ca'].quantile(0.25)
Q3 = hd_df['ca'].quantile(0.75)
IQR = Q3 - Q1
outliers = hd_df[(hd_df['ca'] < Q1 - 1.5 * IQR) | (hd_df['ca'] > Q3 + 1.5 * IQR)]
#plt.figure()
#sns.scatterplot(
#    x='ca',
#    y='num',
#    data = outliers,
#    size=8,
#    palette={0: "green", 1: "red"}
#)
#plt.title("Outliers of ca and presence of CVD")
#plt.xlabel("ca")
#plt.ylabel("num")
#plt.show()

# Handling missing data

print("Missing Values After Preprocessing: \n", hd_df.isnull().sum())

imputer = KNNImputer(n_neighbors=5)
hd_imputed = pd.DataFrame(imputer.fit_transform(hd_encoded), columns=hd_encoded.columns)
print("Missing Values After Imputation: \n", hd_imputed.isnull().sum())

scaler = StandardScaler()
hd_imputed[num_features] = scaler.fit_transform(hd_imputed[num_features])
print("Scaled Numeric Features: \n", hd_imputed[num_features].head())

# Exploratory Analysis with K-Nearest Neighbours (KNN)
X = hd_imputed.drop(columns=['num'])
y = hd_imputed['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'n_neighbors': range(1, 21)}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid=params, cv=5)
grid.fit(X_train, y_train)
best_k = grid.best_params_['n_neighbors']
print(f"Best k: {best_k}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_knn = knn_final.predict(X_test)

print("KNN Accuracy: ", accuracy_score(y_test, y_pred_knn))
print("KNN Confusion Matrix: \n", confusion_matrix(y_test, y_pred_knn))
print("KNN Classification Report: \n", classification_report(y_test, y_pred_knn))

gnb_sel = GaussianNB()
gnb_sel.fit(X_train, y_train)
y_pred_nb = gnb_sel.predict(X_test)

print("Naive Bayes Accuracy: ", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Confusion Matrix: \n", confusion_matrix(y_test, y_pred_nb))
print("Naive Bayes Classification Report: \n", classification_report(y_test, y_pred_nb))

# PCA
X_pca = X.copy()
pca_scaler = StandardScaler()
X_pca_scaled = pca_scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_pca_scaled)
explained_variance = pca.explained_variance_ratio_

print("PCA Explained Variance 2 components: \n", explained_variance)
print("Total Explained Variance: ", explained_variance.sum())

#plt.figure(figsize=(8, 6))
#plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y, cmap='RdYlGn_r', edgecolor='k', s=50)
#plt.title('PCA of Heart Disease Dataset')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.colorbar(label='Heart Disease Presence')
#plt.show()

# Non-Linear Dimensionality Reduction 
X_scaled = X_pca_scaled
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

#plt.figure(figsize=(8, 6))
#sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', edgecolor='k', s=50)
#plt.title('t-SNE of Heart Disease Dataset')
#plt.xlabel('t-SNE Dimension 1')
#plt.ylabel('t-SNE Dimension 2')
#plt.legend(title='Heart Disease Presence')
#plt.show()

umap_reducer = umap.UMAP(n_components=2)
X_umap = umap_reducer.fit_transform(X_scaled)

#plt.figure(figsize=(8, 6))
#sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='coolwarm', alpha=0.7)
#plt.title('UMAP of Heart Disease Dataset')
#plt.xlabel('UMAP Dimension 1')
#plt.ylabel('UMAP Dimension 2')
#plt.legend(title='Heart Disease Presence')
#plt.show()

# Feature Selection and Model Re-evaluation
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)


# Select top 10 features
print("Feature Importances from Random Forest: \n", feat_imp_df.head(8))

selector = RFE(rf, n_features_to_select=8, step=1)
selector.fit(X_train, y_train)
selected_features = X.columns[selector.support_]
print("Selected Features: \n", list(selected_features))

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]
knn_sel = KNeighborsClassifier(n_neighbors=best_k)
knn_sel.fit(X_train_sel, y_train)
y_pred_knn_sel = knn_sel.predict(X_test_sel)

print("KNN with Selected Features Accuracy: ", accuracy_score(y_test, y_pred_knn_sel))
print("KNN Classification Report with Selected Features: \n", classification_report(y_test, y_pred_knn_sel))

gnb_sel.fit(X_train_sel, y_train)
y_pred_nb_sel = gnb_sel.predict(X_test_sel)

print("Naive Bayes Accuracy: ", accuracy_score(y_test, y_pred_nb_sel))
print("Naive Bayes Confusion Matrix: \n", confusion_matrix(y_test, y_pred_nb_sel))
print("Naive Bayes Classification Report: \n", classification_report(y_test, y_pred_nb_sel))


# Additional Models
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sel, y_train)

svc = SVC(probability=True, random_state=42)
svc.fit(X_train_sel, y_train)

#Ensemble Testing
ensemble_all = VotingClassifier(estimators=[
    ('knn', knn_sel),
    ('gnb', gnb_sel),
    ('lr', lr),
    ('svm', svc),
    ('rf', rf)
], voting='soft')

ensemble_all.fit(X_train_sel, y_train)

y_pred_ensemble_all = ensemble_all.predict(X_test_sel)
print("Ensemble Model Accuracy: ", accuracy_score(y_test, y_pred_ensemble_all))

ensemble = VotingClassifier(estimators=[
    ('knn', knn_sel),
    ('gnb', gnb_sel),
    ('lr', lr),
    ('svm', svc),
    ('rf', rf)
], voting='soft')

ensemble.fit(X_train_sel, y_train)

y_pred_ensemble = ensemble.predict(X_test_sel)
print("Ensemble Model Accuracy: ", accuracy_score(y_test, y_pred_ensemble))

ensemble = VotingClassifier(estimators=[
    ('knn', knn_sel),
    ('gnb', gnb_sel),
    ('lr', lr),
    ('svm', svc),
    ('rf', rf)
], voting='soft')

ensemble.fit(X_train_sel, y_train)

y_pred_ensemble = ensemble.predict(X_test_sel)
print("Ensemble Model Accuracy: ", accuracy_score(y_test, y_pred_ensemble))
print("Ensemble Model Classification Report: \n", classification_report(y_test, y_pred_ensemble))

# ROC and Precision-Recall Curves
y_probs = ensemble.predict_proba(X_test_sel)[:, 1]
fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve - Ensemble Classifier")
plt.legend(loc='lower right')
plt.show()

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve - Ensemble Classifier")
plt.legend(loc='lower left')
plt.show()

