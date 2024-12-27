import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import optuna
import datetime

test_split = 0.1
n_splits = 10
n_jobs = 8
n_trials = 75

print(f"Start time: {datetime.datetime.now()}")

# Load the prepared dataset
df = pd.read_csv("csv/MyGamesPrepared.csv")

# Drop specific columns
columns_to_drop = ['Account', 'Color', 'TimeControl', 
                   'ICastledFirst', 'ICastledShort', 'ICastledLong', 'OppCastledShort', 'OppCastledLong',
                   'MyNumMoves', 'OppNumMoves', 'MyTotalTime', 'OppTotalTime', 'MyAvgTPM', 'OppAvgTPM']

df = df.drop(columns=columns_to_drop)

# Separate features and target
X = df.drop(columns=['Result'])
y = df['Result']

# Split into train-test sets for validation during hyperparameter tuning
# Calculate the split point for the last 10%
split_idx = int(len(X) * (1 - test_split))

# Split the data chronologically - last 10% for testing
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Define class weights inversely proportional to their frequencies
# class_weights = {
#     0: 2916 / 1459, # Win
#     1: 2916 / 127,  # Draw
#     2: 2916 / 1330  # Loss
# }

class_weights = {
    0: 100, # Win
    1: 1,  # Draw
    2: 100  # Loss
}

# Define an objective function for Optuna
def objective(trial):
    # Hyperparameter space
    param = {
        'objective': 'multi:softprob',  # Multiclass classification with probabilities
        'num_class': 3,  # Classes: win, draw, loss
        'n_jobs': n_jobs,
        'tree_method': trial.suggest_categorical('tree_method', ['hist']),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    }

    # Initialize the XGBoost model
    model = xgb.XGBClassifier(**param, random_state=999)

    # Perform 10-fold cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=999)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_log_loss')

    # Return the mean log loss (negative because cross_val_score minimizes)
    return -scores.mean()

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Add weights to the training data
sample_weights = y_train.map(class_weights)

# Train the model with the best hyperparameters on the full training set
best_params = study.best_params
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = 3
best_model = xgb.XGBClassifier(**best_params, random_state=999)
best_model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate on the test set
y_pred_proba = best_model.predict_proba(X_test)
y_pred = best_model.predict(X_test)

# Calculate metrics
logloss = log_loss(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

print(f"Log Loss on test set: {logloss}")
print(f"Accuracy on test set: {accuracy}")

# Save the model to JSON format
best_model.get_booster().save_model("json/model.json")
print("Model saved as model.json")

# Display probabilities for the first few test samples
result_probabilities = pd.DataFrame(y_pred_proba, columns=['Win', 'Draw', 'Loss'])
print(result_probabilities.head())

# Plot feature importance
def plot_feature_importance(model, feature_names, importance_type):
    # Extract feature importance
    importance = model.get_booster().get_score(importance_type=importance_type)
    # Convert to DataFrame for easier plotting
    importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
    importance_df.index = feature_names  # Set feature names
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))  # Larger figure size for better readability
    importance_df['Importance'].plot(kind='bar')
    plt.title(f"Feature Importance ({importance_type})")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'png/feature_importance_{importance_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Feature names (columns in the input data)
feature_names = X_train.columns

# Plot importance for 'weight', 'gain', and 'cover'
for importance_type in ['weight', 'gain', 'cover']:
    plot_feature_importance(best_model, feature_names, importance_type=importance_type)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

# Display the confusion matrix as a heatmap
# Display the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('png/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=['Win', 'Draw', 'Loss'])
print(report)

print(f"End time: {datetime.datetime.now()}")