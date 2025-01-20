import warnings
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from sklearn import metrics, model_selection

# Configure settings
warnings.filterwarnings('ignore')  # Suppress warnings
plt.switch_backend('Agg')  # Use non-interactive backend

test_split = 0.1
n_splits = 10
n_trials = 25

print(f"Start time: {datetime.datetime.now()}")

# Load the prepared dataset
df = pd.read_csv("csv/MyGamesPrepared.csv")

df = df[['Result', 'GameOfDay', 'GameOfWeek', 'TimeOfDay', 'TimeSinceLast',
         'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday',
         'DailyWinPerc', 'DailyDrawPerc', 'DailyLossPerc', 'WeeklyWinPerc', 'WeeklyDrawPerc', 'WeeklyLossPerc',
         'Color', 'ECO_A00', 'ECO_A40', 'ECO_A45', 'ECO_B10', 'ECO_B12', 'ECO_B13', 'ECO_D00', 'ECO_D02', 'ECO_D10', 'ECO_Other']]

# Maybe good: 'GameOfDay', 'GameOfWeek', 'DailyWinPerc', 'DailyDrawPerc', 'DailyLossPerc', 'WeeklyWinPerc', 'WeeklyDrawPerc', 'WeeklyLossPerc'

# Drop specific columns
# columns_to_drop = ['Account', 'Color', 'TimeControl', 
#                    'ICastledFirst', 'ICastledShort', 'ICastledLong', 'OppCastledShort', 'OppCastledLong',
#                    'MyNumMoves', 'OppNumMoves', 'MyTotalTime', 'OppTotalTime', 'MyAvgTPM', 'OppAvgTPM',
#                    'TimeOfDay',
#                    'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday']

# df = df.drop(columns=columns_to_drop)

df['TimeSinceLast'] = np.log10(df['TimeSinceLast'] + 1)

# Separate features and target
X = df.drop(columns=['Result'])
y = df['Result']

# Split into train-test sets for validation during hyperparameter tuning
split_idx = int(len(X) * (1 - test_split))

# Split the data chronologically - last 10% for testing
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Convert to DMatrix format for better GPU performance
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define class weights inversely proportional to their frequencies
class_weights = {
    0: 100 * 2916 / 1459, # Win
    1: 1 * 2916 / 127,    # Draw
    2: 100 * 2916 / 1330  # Loss
}

# Define an objective function for Optuna
def objective(trial):
    # Hyperparameter space
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'tree_method': 'hist',
        'device': 'cuda:0',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 7)
    }

    # Create k-fold data
    kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=999)
    scores = []

    for train_idx, valid_idx in kf.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        # Convert to DMatrix
        d_fold_train = xgb.DMatrix(X_fold_train, label=y_fold_train)
        d_fold_valid = xgb.DMatrix(X_fold_valid, label=y_fold_valid)

        # Train model
        model = xgb.train(
            param,
            d_fold_train,
            num_boost_round=1000,
            evals=[(d_fold_valid, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predict and compute log loss
        pred = model.predict(d_fold_valid)
        score = metrics.log_loss(y_fold_valid, pred)
        scores.append(score)

    return sum(scores) / len(scores)

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials)

# Print the best hyperparameters
print("\nBest hyperparameters:", study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params
best_params.update({
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'device': 'cuda:0'
})

# Add sample weights to the DMatrix
sample_weights = y_train.map(class_weights)
dtrain.set_weight(sample_weights)

# Train the final model
best_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=False
)

# Evaluate on the test set
y_pred_proba = best_model.predict(dtest)
y_pred = y_pred_proba.argmax(axis=1)

# Calculate metrics
logloss = metrics.log_loss(y_test, y_pred_proba)
accuracy = metrics.accuracy_score(y_test, y_pred)

print(f"\nLog Loss on test set: {logloss:.4f}")
print(f"Accuracy on test set: {accuracy:.4f}")

# Save the model
best_model.save_model("json/model.json")
print("Model saved as model.json")

# Plot feature importance
def plot_feature_importance(model, feature_names, importance_type):
    importance = model.get_score(importance_type=importance_type)
    importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
    
    # Only use feature names that appear in the importance scores
    common_features = [feat for feat in feature_names if feat in importance_df.index]
    if not common_features:
        print(f"No features found with non-zero {importance_type} importance")
        return
        
    importance_df = importance_df.loc[common_features]
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    importance_df['Importance'].plot(kind='bar')
    plt.title(f"Feature Importance ({importance_type})")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'png/feature_importance_{importance_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Feature names
feature_names = X_train.columns

# Plot importance for different metrics
for importance_type in ['weight', 'gain', 'cover']:
    plot_feature_importance(best_model, feature_names, importance_type=importance_type)

# Compute and plot confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Win', 'Draw', 'Loss'])
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('png/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate classification report
report = metrics.classification_report(y_test, y_pred, target_names=['Win', 'Draw', 'Loss'])
print("\nClassification Report:")
print(report)

print(f"End time: {datetime.datetime.now()}")