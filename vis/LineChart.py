import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Read the CSV file
df = pd.read_csv('csv/MyGamesCombined.csv')

# Convert the "Date" column to datetime format for filtering
df['Date'] = pd.to_datetime(df['Date'])

# Filter the data for "Cosmos_IV" with TimeControl == 600 and Date >= May 31, 2024
cosmos_iv_data = df[
    (df['Account'] == 'Cosmos_IV') & 
    (df['TimeControl'] == 600) & 
    (df['Date'] >= '2024-05-31')
]

# Initialize variables for tracking ATH and color values
ath = -float('inf')  # Start with a very low value
colors = []  # Store colors for each point

# Iterate through "GameNumber" in ascending order and determine colors
for elo in cosmos_iv_data['MyElo']:
    if elo > ath:
        ath = elo  # Update ATH
        colors.append('#FFD700')  # Color ATH as gold
    else:
        percentage_decrease = (ath - elo) / ath * 100  # Calculate % decrease from ATH
        percentage_decrease = min(percentage_decrease, 15)  # Cap percentage decrease at 15%

        # Normalize percentage decrease for color gradient
        norm = mcolors.Normalize(vmin=0, vmax=15)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "ath_gradient", ['#0066FF', '#FF3333']
        )
        colors.append(cmap(norm(percentage_decrease)))  # Map percentage to gradient color

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(30, 8))

# Set dark background
fig.patch.set_facecolor('#1E1E1E')
ax.set_facecolor('#1E1E1E')

# Scatter plot Elo progression with custom colors
ax.scatter(
    cosmos_iv_data['GameNumber'], 
    cosmos_iv_data['MyElo'], 
    c=colors, 
    s=50,  # Size of points
    edgecolor='none'
)

# Customize the chart
ax.set_title('My Elo Progression (10 Minute Games, Post-May 31, 2024)', 
             color='white', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Game Number', color='white', fontsize=14, fontweight='bold')
ax.set_ylabel('Elo', color='white', fontsize=14, fontweight='bold')

# Style x and y axis
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Remove borders (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

# Add a colorbar to show the percentage decrease
import matplotlib.cm as cm
cbar = plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap), 
    ax=ax, 
    orientation='vertical', 
    pad=0.02
)
cbar.set_label('% Decrease from ATH', color='white', fontsize=14, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Adjust layout and save the chart with dark background
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig('png/vis/Progression.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='#1E1E1E', 
            edgecolor='none')

plt.close()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Read the CSV file
df = pd.read_csv('csv/MyGamesCombined.csv')

# Convert the "Date" column to datetime format for filtering
df['Date'] = pd.to_datetime(df['Date'])

# Filter the data for "Cosmos_IV" with TimeControl == 600 and Date >= May 31, 2024
cosmos_iv_data = df[
    (df['Account'] == 'Cosmos_IV') & 
    (df['TimeControl'] == 600) & 
    (df['Date'] >= '2024-05-31')
]

# Initialize variables for tracking ATH and color values
ath = -float('inf')  # Start with a very low value
colors = []  # Store colors for each point

# Iterate through "GameNumber" in ascending order and determine colors
for elo in cosmos_iv_data['MyElo']:
    if elo > ath:
        ath = elo  # Update ATH
        colors.append('#FFD700')  # Color ATH as gold
    else:
        percentage_decrease = (ath - elo) / ath * 100  # Calculate % decrease from ATH
        percentage_decrease = min(percentage_decrease, 15)  # Cap percentage decrease at 15%

        # Normalize percentage decrease for color gradient
        vmax = 100/7

        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "ath_gradient", ['#0066FF', '#FF3333']
        )
        colors.append(cmap(norm(percentage_decrease)))  # Map percentage to gradient color

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(30, 8))

# Set dark background
fig.patch.set_facecolor('#1E1E1E')
ax.set_facecolor('#1E1E1E')

# Scatter plot Elo progression with custom colors
ax.scatter(
    cosmos_iv_data['GameNumber'], 
    cosmos_iv_data['MyElo'], 
    c=colors, 
    s=50,  # Size of points
    edgecolor='none'
)

# Customize the chart
ax.set_title('My Elo Progression (10 Minute Games, Post-May 31, 2024)', 
             color='white', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Game Number', color='white', fontsize=14, fontweight='bold')
ax.set_ylabel('Elo', color='white', fontsize=14, fontweight='bold')

# Style x and y axis
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Remove borders (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

# Add a colorbar to show the percentage decrease
import matplotlib.cm as cm
cbar = plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap), 
    ax=ax, 
    orientation='vertical', 
    pad=0.02
)
cbar.set_label('% Decrease from ATH', color='white', fontsize=14, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Adjust layout and save the chart with dark background
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig('png/vis/Progression.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='#1E1E1E', 
            edgecolor='none')

plt.close()
