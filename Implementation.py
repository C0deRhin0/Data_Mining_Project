# %% [markdown]
# 1. Import Libraries

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import time
import scipy.stats as stats

# %% [markdown]
# 2. Data Loading

# %%
try:
    train = pd.read_csv('Datasets/UNSW_NB15_training-set.csv')
    test = pd.read_csv('Datasets/UNSW_NB15_testing-set.csv')
    print("\nData loaded successfully")
    print(f"\nTraining set shape: {train.shape}")
    print(f"\nTesting set shape: {test.shape}")
    
    # Initial data check
    print("\nTraining set preview:")
    display(train.head())
    print("\nTesting set preview:")
    display(test.head())
    
except Exception as e:
    print(f"Data loading failed: {e}")

# %% [markdown]
# 3. Data Preprocessing

# %%
# Combine data
df = pd.concat([train, test])
print("\n🔍 Combined dataset shape:", df.shape)

# Drop duplicates
df = df.drop_duplicates()
print(f"\n✅ Shape after removing duplicates: {df.shape}")

# Drop irrelevant features
irrelevant_cols = ['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'attack_cat', 'id', 'ttl']
df = df.drop(columns=irrelevant_cols, errors='ignore')
print("\n✅ Irrelevant columns removed")
print("Remaining columns:", df.columns.tolist())
print("\n🔍 Data after column removal:")
display(df.head(2))

# Handle missing values
print("\n🔍 Missing values before handling:")
print(df.isna().sum().sort_values(ascending=False).head())

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\n✅ Missing values handled")
print("🔍 Missing values after handling:")
print(df.isna().sum().sort_values(ascending=False).head())

# Feature Encoding
label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])
    
print("\n✅ Categorical features encoded")
print("\n🔍 Encoded data preview:")
display(df[cat_cols].head(2))

# Normalization
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\n✅ Numerical features normalized")
print("\n🔍 Normalized data preview:")
display(df[num_cols].head(2))

# %% [markdown]
# 4. Feature Selection

# %%
X = df.drop(columns=['label'])
y = df['label']
original_features = X.shape[1]

# Stage 1: Filter-Based Ranking
# Information Gain
info_gain = mutual_info_classif(X, y, random_state=42)
info_gain_rank = pd.Series(info_gain, index=X.columns).sort_values(ascending=False)
print("\n🔍 Top 5 features by Information Gain:")
print(info_gain_rank.head())

# Random Forest Importance
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
rf.fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🔍 Top 5 features by RF Importance:")
print(rf_importance.head())

# Hybrid ranking
hybrid_rank = (info_gain_rank + rf_importance).sort_values(ascending=False)
top_features = hybrid_rank.head(30).index.tolist()
print("\n🔍 Top 10 hybrid-ranked features:")
print(hybrid_rank.head(10))

# Stage 2: RFECV
estimator = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
selector = RFECV(estimator, step=1, cv=5, scoring='f1', n_jobs=-1, min_features_to_select=16)
selector.fit(X[top_features], y)
selected_features = X[top_features].columns[selector.support_]
print(f"\n✅ Final selected features: {selector.n_features_}")
print(f"Reduction Rate: {((original_features - selector.n_features_)/original_features)*100:.1f}%")
print(selected_features.tolist())

# %% [markdown]
# 5. Model Training

# %%
X_final = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, stratify=y, random_state=42)
print("\n🔍 Data split shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("\n✅ Best parameters:", grid_search.best_params_)

# %% [markdown]
# 6. Evaluation

# %%
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📊 Final Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Dimensionality Reduction Rate
final_features = X_final.shape[1]
print(f"\n📉 Dimensionality Reduction: {original_features} → {final_features} features")
print(f"Reduction Rate: {((original_features - final_features)/original_features)*100:.1f}%")

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
pd.Series(best_model.feature_importances_, index=selected_features)\
  .sort_values()\
  .tail(15)\
  .plot(kind='barh', title='Top 15 Important Features')
plt.show()

# %% [markdown]
# 7. Enhanced Evaluation

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n🔴🟢 Confusion Matrix:")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

# Classification Report
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Cohen's Kappa
kappa = cohen_kappa_score(y_test, y_pred)
print(f"\n📐 Cohen's Kappa Score: {kappa:.4f}")

# Cross-Validation Scores
cv_scores = cross_val_score(best_model, X_final, y, cv=5, scoring='f1', n_jobs=-1)
print(f"\n🎯 5-Fold Cross-Validation F1 Scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Statistical Significance
t_stat, p_value = stats.ttest_1samp(cv_scores, 0.95)
print(f"\n📈 T-Test vs Target F1=0.95: t={t_stat:.2f}, p={p_value:.4f}")

# Fit Time
start_time = time.time()
best_model.fit(X_train, y_train)
fit_time = time.time() - start_time
print(f"\n⏱ Model Fit Time: {fit_time:.2f} seconds")

# Goal Achievement Check
target_accuracy = 0.95
target_f1 = 0.95
achieved_accuracy = accuracy_score(y_test, y_pred)
achieved_f1 = f1_score(y_test, y_pred)
print("\n🎯 Goal Achievement Status:")
print(f"Accuracy Target ({target_accuracy*100}%): ({achieved_accuracy:.4f})")
print(f"F1-Score Target ({target_f1}): ({achieved_f1:.4f})")
print(f"Feature Reduction: {((original_features - final_features)/original_features)*100:.1f}% (Target: 50-70%)")

# %% [markdown]
# 8. Feature Ablation Study

# %%
# Identify top features based on Random Forest importance
feature_importance = pd.Series(best_model.feature_importances_, index=selected_features).sort_values(ascending=False)
top_features = feature_importance.head(2).index.tolist()
print("\n🔍 Top 2 Features for Ablation:")
print(top_features)

# Perform ablation
ablation_results = []
for feature in top_features + [top_features]:  # Individual and combined ablation
    if isinstance(feature, list):
        ablation_name = "Top 2 Features Removed"
        X_train_ablated = X_train.drop(columns=feature)
        X_test_ablated = X_test.drop(columns=feature)
    else:
        ablation_name = f"{feature} Removed"
        X_train_ablated = X_train.drop(columns=[feature])
        X_test_ablated = X_test.drop(columns=[feature])
    
    # Retrain model
    model_ablated = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=42)
    start_time = time.time()
    model_ablated.fit(X_train_ablated, y_train)
    fit_time_ablated = time.time() - start_time
    y_pred_ablated = model_ablated.predict(X_test_ablated)
    
    # Compute metrics
    metrics = {
        'Ablation': ablation_name,
        'Accuracy': accuracy_score(y_test, y_pred_ablated),
        'F1-Score': f1_score(y_test, y_pred_ablated),
        'Precision': precision_score(y_test, y_pred_ablated),
        'Recall': recall_score(y_test, y_pred_ablated),
        'Fit Time (s)': fit_time_ablated
    }
    ablation_results.append(metrics)

# Baseline metrics for comparison
baseline_metrics = {
    'Ablation': 'Baseline (No Ablation)',
    'Accuracy': achieved_accuracy,
    'F1-Score': achieved_f1,
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Fit Time (s)': fit_time
}
ablation_results.append(baseline_metrics)

# Display results
ablation_df = pd.DataFrame(ablation_results)
print("\n📊 Feature Ablation Results:")
display(ablation_df)

# Visualize F1-Score changes
plt.figure(figsize=(8, 5))
sns.barplot(x='F1-Score', y='Ablation', data=ablation_df)
plt.title('F1-Score Across Feature Ablation Scenarios')
plt.axvline(x=target_f1, color='red', linestyle='--', label='Target F1 (0.95)')
plt.legend()
plt.show()

# %% [markdown]
# 9. Noise Injection Study

# %%
# Create noisy datasets
noise_ratio = 0.05  # 5% noise
X_train_noisy = X_train.copy().astype(np.float64)  # Ensure float64 dtype
X_test_noisy = X_test.copy().astype(np.float64)

# Perturb 5% of values with Gaussian noise
for col in X_train_noisy.columns:
    mask = np.random.choice([True, False], size=X_train_noisy.shape[0], p=[noise_ratio, 1-noise_ratio])
    X_train_noisy.loc[mask, col] = X_train_noisy.loc[mask, col] + np.random.normal(0, 0.1, sum(mask))
    X_train_noisy[col] = np.clip(X_train_noisy[col], 0, 1)  # Ensure [0, 1] range

    mask_test = np.random.choice([True, False], size=X_test_noisy.shape[0], p=[noise_ratio, 1-noise_ratio])
    X_test_noisy.loc[mask_test, col] = X_test_noisy.loc[mask_test, col] + np.random.normal(0, 0.1, sum(mask_test))
    X_test_noisy[col] = np.clip(X_test_noisy[col], 0, 1)

# Drop 5% of values (set to NaN and fill with median)
X_train_missing = X_train.copy().astype(np.float64)
X_test_missing = X_test.copy().astype(np.float64)
for col in X_train_missing.columns:
    mask = np.random.choice([True, False], size=X_train_missing.shape[0], p=[noise_ratio, 1-noise_ratio])
    X_train_missing.loc[mask, col] = np.nan
    X_train_missing[col] = X_train_missing[col].fillna(X_train[col].median())

    mask_test = np.random.choice([True, False], size=X_test_missing.shape[0], p=[noise_ratio, 1-noise_ratio])
    X_test_missing.loc[mask_test, col] = np.nan
    X_test_missing[col] = X_test_missing[col].fillna(X_test[col].median())

# Train and evaluate on noisy data
noise_scenarios = [
    ('Gaussian Noise (5%)', X_train_noisy, X_test_noisy),
    ('Missing Values (5%)', X_train_missing, X_test_missing)
]
noise_results = []

for scenario, X_train_sc, X_test_sc in noise_scenarios:
    model_noise = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=42)
    start_time = time.time()
    model_noise.fit(X_train_sc, y_train)
    fit_time_noise = time.time() - start_time
    y_pred_noise = model_noise.predict(X_test_sc)
    
    metrics = {
        'Scenario': scenario,
        'Accuracy': accuracy_score(y_test, y_pred_noise),
        'F1-Score': f1_score(y_test, y_pred_noise),
        'Precision': precision_score(y_test, y_pred_noise),
        'Recall': recall_score(y_test, y_pred_noise),
        'Fit Time (s)': fit_time_noise
    }
    noise_results.append(metrics)

# Add baseline
noise_results.append(baseline_metrics.copy())
noise_results[-1]['Scenario'] = 'Baseline (No Noise)'

# Display results
noise_df = pd.DataFrame(noise_results)
print("\n📊 Noise Injection Results:")
display(noise_df)

# Visualize F1-Score changes
plt.figure(figsize=(8, 5))
sns.barplot(x='F1-Score', y='Scenario', data=noise_df)
plt.title('F1-Score Across Noise Injection Scenarios')
plt.axvline(x=target_f1, color='red', linestyle='--', label='Target F1 (0.95)')
plt.legend()
plt.show()

# %% [markdown]
# 10. Alternate Models Benchmarking

# %%
# Alternate Feature Selection: SelectKBest
selector_kbest = SelectKBest(score_func=mutual_info_classif, k=16)
X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
X_test_kbest = selector_kbest.transform(X_test)
kbest_features = X_train.columns[selector_kbest.get_support()].tolist()
print("\n🔍 SelectKBest Features (16):")
print(kbest_features)

# Alternate Classifiers
alt_models = [
    ('SelectKBest + RF', RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=42), X_train_kbest, X_test_kbest),
    ('RFECV + SVM', SVC(kernel='rbf', class_weight='balanced', random_state=42), X_train, X_test),  # Note: SVC does not support n_jobs
    ('RFECV + XGBoost', XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, eval_metric='logloss', random_state=42), X_train, X_test)
]

alt_results = []
for name, model, X_tr, X_te in alt_models:
    start_time = time.time()
    model.fit(X_tr, y_train)
    fit_time_alt = time.time() - start_time
    start_time = time.time()
    y_pred_alt = model.predict(X_te)
    inf_time_alt = time.time() - start_time
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred_alt),
        'F1-Score': f1_score(y_test, y_pred_alt),
        'Precision': precision_score(y_test, y_pred_alt),
        'Recall': recall_score(y_test, y_pred_alt),
        'Fit Time (s)': fit_time_alt,
        'Inference Time (s)': inf_time_alt
    }
    alt_results.append(metrics)

# Add baseline
alt_results.append({
    'Model': 'RFECV + RF (Baseline)',
    'Accuracy': achieved_accuracy,
    'F1-Score': achieved_f1,
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Fit Time (s)': fit_time,
    'Inference Time (s)': fit_time / len(X_test)  # Approximate
})

# Display results
alt_df = pd.DataFrame(alt_results)
print("\n📊 Alternate Models Benchmarking Results:")
display(alt_df)

# Visualize F1-Score comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='F1-Score', y='Model', data=alt_df)
plt.title('F1-Score Across Alternate Models')
plt.axvline(x=target_f1, color='red', linestyle='--', label='Target F1 (0.95)')
plt.legend()
plt.show()

# Visualize Fit Time comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Fit Time (s)', y='Model', data=alt_df)
plt.title('Fit Time Across Alternate Models')
plt.show()
