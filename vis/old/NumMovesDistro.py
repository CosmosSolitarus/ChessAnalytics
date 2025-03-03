import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('csv/MyGamesCombined.csv')

# Filter data for "Cosmos_IV" with TimeControl = 600
filtered_data = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]

# Separate data by result
wins = filtered_data[filtered_data['Result'] == 'won']
losses = filtered_data[filtered_data['Result'] == 'lost']
draws = filtered_data[filtered_data['Result'] == 'draw']

# Plot overlapping KDEs for "MyNumMoves"
plt.figure(figsize=(12, 8))

sns.kdeplot(wins['MyNumMoves'], fill=True, alpha=0.6, color='#137d3f', label='Wins')
sns.kdeplot(losses['MyNumMoves'], fill=True, alpha=0.6, color='#d62728', label='Losses')
sns.kdeplot(draws['MyNumMoves'], fill=True, alpha=0.6, color='#1f77b4', label='Draws')

# Customize the chart
plt.title('Number of Moves by Game Result', fontsize=16, fontweight='bold')
plt.xlabel('Number of Moves (MyNumMoves)', fontsize=14, fontweight='bold')
plt.ylabel('Density', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Customize legend
plt.legend(
    title='Result',
    fontsize=14,
    title_fontsize=16,
    frameon=False,
    prop={'weight': 'bold'}
)

# Remove borders (spines)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Display the chart
plt.tight_layout()
plt.show()
