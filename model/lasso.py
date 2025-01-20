import cudf
import cupy as cp
import numpy as np
from itertools import combinations
import time
from sklearn.model_selection import train_test_split, LeaveOneOut
from cuml.linear_model import LogisticRegression
from cuml.metrics import log_loss
from cuml.preprocessing import StandardScaler as cuStandardScaler

start_time = time.time()

# Load and prepare data
df = cudf.read_csv('csv/MyGamesPrepared.csv')
df = df.dropna()

# Define column types
cat_columns = ['IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 
               'IsSaturday', 'IsSunday', 'Color', 'ECO_A00', 'ECO_A40', 'ECO_A45', 
               'ECO_B10', 'ECO_B12', 'ECO_B13', 'ECO_D00', 'ECO_D02', 'ECO_D10', 'ECO_Other']
num_columns = ['GameOfDay', 'GameOfWeek', 'TimeOfDay', 'TimeSinceLast',
               'DailyWinPerc', 'DailyDrawPerc', 'DailyLossPerc', 
               'WeeklyWinPerc', 'WeeklyDrawPerc', 'WeeklyLossPerc']

#df['TimeSinceLast'] = np.log10(df['TimeSinceLast'] + 1)

# Prepare X and y
X = df[num_columns + cat_columns]
y = df['Result']

# Create interactions between numerical features only
X_with_interactions = X.copy()
for col1, col2 in combinations(num_columns, 2):
    interaction_name = f"{col1}_{col2}"
    X_with_interactions[interaction_name] = X[col1] * X[col2]

# Scale numerical features and interactions
scaler = cuStandardScaler()
numerical_cols = num_columns + [col for col in X_with_interactions.columns if '_' in col]
X_with_interactions[numerical_cols] = scaler.fit_transform(X_with_interactions[numerical_cols])

# Train-test split
X_np = X_with_interactions.to_pandas().values
y_np = y.to_pandas().values
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.1, random_state=42, stratify=y_np
)

# Convert back to GPU
X_train = cudf.DataFrame(X_train, columns=X_with_interactions.columns)
X_test = cudf.DataFrame(X_test, columns=X_with_interactions.columns)
y_train = cudf.Series(y_train)
y_test = cudf.Series(y_test)

# Train LASSO model
lasso = LogisticRegression(
    penalty='l1',
    solver='qn',
    l1_ratio=1.0,
    max_iter=1000
)

lasso.fit(X_train, y_train)

# Calculate test set log loss
y_pred_proba = lasso.predict_proba(X_test)
test_log_loss = log_loss(y_test, y_pred_proba)
print(f"Test set log loss: {test_log_loss:.4f}")

# Perform LOOCV
loo = LeaveOneOut()
loo_scores = []

# Convert data to CPU for LOOCV
X_cpu = X_with_interactions.to_pandas()
y_cpu = y.to_pandas()

for train_idx, test_idx in loo.split(X_cpu):
    X_loo_train = cudf.DataFrame(X_cpu.iloc[train_idx].values, columns=X_cpu.columns)
    X_loo_test = cudf.DataFrame(X_cpu.iloc[test_idx].values, columns=X_cpu.columns)
    y_loo_train = cudf.Series(y_cpu.iloc[train_idx].values)
    y_loo_test = cudf.Series(y_cpu.iloc[test_idx].values)
    
    lasso_loo = LogisticRegression(
        penalty='l1',
        solver='qn',
        l1_ratio=1.0,
        max_iter=1000
    )
    lasso_loo.fit(X_loo_train, y_loo_train)
    
    y_loo_pred_proba = lasso_loo.predict_proba(X_loo_test)
    loo_scores.append(float(log_loss(y_loo_test, y_loo_pred_proba)))

print(f"Mean LOOCV log loss: {np.mean(loo_scores):.4f}")
print(f"Std LOOCV log loss: {np.std(loo_scores):.4f}")

# Print feature coefficients
feature_names = X_with_interactions.columns
coef_matrix = lasso.coef_.get()
for class_idx in range(coef_matrix.shape[0]):
    coefficients = coef_matrix[class_idx]
    non_zero_mask = coefficients != 0
    non_zero_features = feature_names[non_zero_mask]
    non_zero_coeffs = coefficients[non_zero_mask]
    print(f"\nClass {class_idx}:")
    for feature, coef in zip(non_zero_features, non_zero_coeffs):
        print(f"{feature}: {coef:.4f}")

# Clear GPU memory
cp.get_default_memory_pool().free_all_blocks()