import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load the XGBoost model from the .json file
model_path = "json/model.json"  # Replace with your model's path
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

input_vector = np.array([[0, 0, 0, 0, 1, 0, 0,  # day of week (Friday)
                           54360,              # time of day (0-86399)
                           3,                  # games today
                           40,                 # games this week
                           0,                  # Elo difference (MyElo-OppElo)
                           1, 0, 0,            # last result (win/loss/draw)
                           0, 0, 1,            # 2nd last result (win/loss/draw)
                           79200]])            # time since last game (in seconds)

# Convert the input vector into a DMatrix for XGBoost
dmatrix = xgb.DMatrix(input_vector, feature_names=feature_names)

# Use SHAP to explain the predictions
explainer = shap.Explainer(xgb_model)
shap_values = explainer(dmatrix)

# Extract the SHAP values and base values
shap_values_array = shap_values.values[0]  # Extract the first set of SHAP values (assuming single prediction)
if isinstance(shap_values_array, np.ndarray) and shap_values_array.ndim > 1:
    shap_values_array = shap_values_array  # Keep the 2D array for multi-class models

base_value = shap_values.base_values[0] if hasattr(shap_values.base_values, "__len__") else shap_values.base_values

# Display SHAP values for each class
print("SHAP values for each feature and class:")
for class_idx, class_shap_values in enumerate(shap_values_array):  # Loop over classes
    print(f"\nClass {class_idx} SHAP values:")
    for feature, value in zip(feature_names, class_shap_values):
        print(f"  {feature}: {value:.4f}")

# Visualize the SHAP values for the first class (e.g., class 0)
for class_to_visualize in range(len(feature_names)):
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_array[class_to_visualize],  # SHAP values for the chosen class
        base_values=base_value[class_to_visualize] if isinstance(base_value, np.ndarray) else base_value, 
        data=input_vector[0],
        feature_names=feature_names
    ))
