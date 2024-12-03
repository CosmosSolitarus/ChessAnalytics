import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import optuna

# Load the prepared dataset
df = pd.read_csv("MyGamesPrepared.csv")

# Separate features and target
X = df.drop(columns=['Result'])
y = df['Result']

# Split into train-test sets for validation during hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999, stratify=y)

# Define an objective function for Optuna
def objective(trial):
    # Hyperparameter space
    param = {
        'objective': 'multi:softprob',  # Multiclass classification with probabilities
        'num_class': 3,  # Classes: win, draw, loss
        'n_jobs': 6,
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }

    # Initialize the XGBoost model
    model = xgb.XGBClassifier(**param, random_state=999)

    # Perform 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=999)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_log_loss')

    # Return the mean log loss (negative because cross_val_score minimizes)
    return -scores.mean()

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train the model with the best hyperparameters on the full training set
best_params = study.best_params
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = 3
best_model = xgb.XGBClassifier(**best_params, random_state=999)
best_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred_proba = best_model.predict_proba(X_test)
y_pred = best_model.predict(X_test)

# Calculate metrics
logloss = log_loss(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

print(f"Log Loss on test set: {logloss}")
print(f"Accuracy on test set: {accuracy}")

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
    plt.figure(figsize=(10, 6))
    importance_df['Importance'].plot(kind='bar')
    plt.title(f"Feature Importance ({importance_type})")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Feature names (columns in the input data)
feature_names = X_train.columns

# Plot importance for 'weight', 'gain', and 'cover'
for importance_type in ['weight', 'gain', 'cover']:
    plot_feature_importance(best_model, feature_names, importance_type=importance_type)