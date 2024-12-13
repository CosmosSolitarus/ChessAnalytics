import xgboost as xgb
import pandas as pd
import numpy as np
import json

def analyze_xgboost_model(model_path, feature_names):
    """
    Comprehensive analysis of XGBoost model feature importance
    
    Parameters:
    - model_path: Path to the saved XGBoost model (JSON)
    - feature_names: List of feature names in the model
    
    Returns:
    - DataFrame with detailed feature importance metrics
    """
    # Load the model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Get feature importance
    importance_gain = model.get_score(importance_type='gain')
    importance_weight = model.get_score(importance_type='weight')
    importance_cover = model.get_score(importance_type='cover')
    
    # Prepare results DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Gain': [importance_gain.get(f, 0) for f in feature_names],
        'Weight': [importance_weight.get(f, 0) for f in feature_names],
        'Cover': [importance_cover.get(f, 0) for f in feature_names]
    })
    
    # Add normalized and percentage columns
    feature_importance_df['Gain_Normalized'] = feature_importance_df['Gain'] / feature_importance_df['Gain'].sum() * 100
    feature_importance_df['Weight_Normalized'] = feature_importance_df['Weight'] / feature_importance_df['Weight'].sum() * 100
    feature_importance_df['Cover_Normalized'] = feature_importance_df['Cover'] / feature_importance_df['Cover'].sum() * 100
    
    # Sort by gain for easy interpretation
    feature_importance_df = feature_importance_df.sort_values('Gain', ascending=False)
    
    return feature_importance_df

def additional_model_insights(model_path):
    """
    Extract additional insights from the XGBoost model
    
    Parameters:
    - model_path: Path to the saved XGBoost model (JSON)
    
    Returns:
    - Dictionary of additional model characteristics
    """
    # Load the model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Extract model parameters
    model_config = json.loads(model.save_config())
    
    insights = {
        'num_trees': len(model.get_dump()),
        'max_depth': model_config.get('learner', {}).get('gradient_booster', {}).get('tree_train_param', {}).get('max_depth', 'N/A'),
        'learning_rate': model_config.get('learner', {}).get('gradient_booster', {}).get('learning_rate', 'N/A')
    }
    
    return insights

# Example usage
feature_names = ['IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday', 
                 'TimeOfDay', 'GameOfDay', 'GameOfWeek', 'EloDifference', 
                 'LastResultIsWin', 'LastResultIsDraw', 'LastResultIsLoss', 
                 '2ndLastResultIsWin', '2ndLastResultIsDraw', '2ndLastResultIsLoss', 
                 'TimeSinceLast']

# Path to your model
model_path = 'json/model.json'

# Analyze feature importance
feature_importance = analyze_xgboost_model(model_path, feature_names)
print("Feature Importance Analysis:")
print(feature_importance)
print("\n")

# Get additional model insights
model_insights = additional_model_insights(model_path)
print("Model Insights:")
print(model_insights)

# Quick visualization of feature importance
def plot_feature_importance(df):
    """
    Create a simple visualization of feature importance
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.bar(df['Feature'], df['Gain_Normalized'])
    plt.title('Feature Importance (Gain)')
    plt.xlabel('Features')
    plt.ylabel('Importance (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

plot_feature_importance(feature_importance)