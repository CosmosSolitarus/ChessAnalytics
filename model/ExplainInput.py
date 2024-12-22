import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load the XGBoost model from the .json file
model_path = "json/model.json"
xgb_model = xgb.Booster()
xgb_model.load_model(model_path)

# Define feature names and input vector
feature_names = [
    'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday',
    'TimeOfDay', 'GameOfDay', 'GameOfWeek', 'EloDifference',
    'LastResultIsWin', 'LastResultIsDraw', 'LastResultIsLoss',
    '2ndLastResultIsWin', '2ndLastResultIsDraw', '2ndLastResultIsLoss',
    'TimeSinceLast'
]

input_vector = np.array([[0, 0, 0, 1, 0, 0, 0,  # day of week (Mon-Sun)
                         41400,              # time of day (0-86399)
                         0,                  # games today
                         40,                 # games this week
                         0,                  # Elo difference (MyElo-OppElo)
                         1, 0, 0,            # last result (win/draw/loss)
                         1, 0, 0,            # 2nd last result (win/draw/loss)
                         20]])            # time since last game (in seconds)

# Convert the input vector into a DMatrix for XGBoost
dmatrix = xgb.DMatrix(input_vector, feature_names=feature_names)

# Create tree explainer for XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Get SHAP values
shap_values = explainer.shap_values(input_vector)

# Get predictions
predictions = xgb_model.predict(dmatrix)

# Print predictions first
print("\nPredicted probabilities:")
print(f"Win:  {predictions[0,0]:.1%}")
print(f"Draw: {predictions[0,1]:.1%}")
print(f"Loss: {predictions[0,2]:.1%}")

# Print feature impacts for each class
class_names = ['Win', 'Draw', 'Loss']
print("\nTop feature impacts by class:")

for class_idx, class_name in enumerate(class_names):
    # Get absolute SHAP values for this class
    class_shap_values = np.abs(shap_values[0,:,class_idx])
    # Sort features by importance
    feature_importance = list(zip(feature_names, class_shap_values))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{class_name} probability ({predictions[0,class_idx]:.1%}) impacted most by:")
    for feature, impact in feature_importance[:5]:  # Top 5 features
        raw_impact = shap_values[0,feature_names.index(feature),class_idx]
        direction = "+" if raw_impact > 0 else "-"
        print(f"{feature}: {direction}{abs(raw_impact):.4f}")

# Create separate bar plots for each class
for class_idx, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 6))
    
    # Get SHAP values for this class
    class_values = shap_values[0,:,class_idx]
    
    # Sort by absolute value
    importance_order = np.argsort(np.abs(class_values))
    
    # Create bar plot
    plt.barh(
        [feature_names[i] for i in importance_order],
        [class_values[i] for i in importance_order],
        color=['red' if x < 0 else 'blue' for x in [class_values[i] for i in importance_order]]
    )
    
    plt.title(f'Feature Impact on {class_name} Probability')
    plt.xlabel('SHAP Value (negative = decreases probability, positive = increases)')
    
    plt.tight_layout()
    plt.savefig(f'shap_impact_{class_name.lower()}.png')
    plt.close()

# Create a heatmap of all SHAP values
plt.figure(figsize=(12, 8))
shap_data = shap_values[0]

plt.imshow(shap_data, aspect='auto')
plt.colorbar(label='SHAP value')
plt.xticks(range(3), class_names)
plt.yticks(range(len(feature_names)), feature_names)
plt.title('Feature Impact Across All Outcomes')

plt.tight_layout()
plt.savefig('shap_heatmap_all.png')
plt.close()