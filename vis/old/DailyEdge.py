import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Read the CSV file
df = pd.read_csv('csv/MyGamesCombined.csv')

# Filter data for "Cosmos_IV" with TimeControl = 600
filtered_data = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]

# Group data by "DayOfWeek" and "Result", then calculate the counts
daily_results = filtered_data.groupby(['DayOfWeek', 'Result']).size().unstack(fill_value=0)

# Calculate percentages for wins and losses
daily_percentages = daily_results.div(daily_results.sum(axis=1), axis=0) * 100

# Calculate the "edge" as win% - loss%
daily_edge = daily_percentages.get('won', 0) - daily_percentages.get('lost', 0)

# Rename index to three-letter abbreviations for days
day_map = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
daily_edge.index = [day_map[idx] for idx in daily_edge.index]

# Calculate overall edge across all games
overall_edge = (filtered_data['Result'].value_counts(normalize=True)['won'] -
                filtered_data['Result'].value_counts(normalize=True)['lost']) * 100

# Calculate 99% confidence intervals for each day's edge
n_games_per_day = daily_results.sum(axis=1)
z_score = stats.norm.ppf(0.995)  # 99% confidence level
ci_half_width = z_score * np.sqrt((daily_percentages.get('won', 0) / 100) *
                                  (1 - daily_percentages.get('won', 0) / 100) /
                                  n_games_per_day)

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(14, 8))  # Extended width for label placement

# Set dark background
fig.patch.set_facecolor('#1E1E1E')
ax.set_facecolor('#1E1E1E')

# Plot the edge as a bar chart
bars = ax.bar(daily_edge.index, daily_edge, color='#4fa4f7', edgecolor='none')

# Add data labels
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.1f}',
        ha='center',
        va='bottom' if height >= 0 else 'top',
        color='white',
        fontweight='bold'
    )

# Add a horizontal dashed line for overall edge
ax.axhline(y=overall_edge, color='lightgrey', linestyle='--', linewidth=1.5)
ax.text(
    len(daily_edge),
    overall_edge,
    f'Overall Edge: {overall_edge:.1f}%',
    color='lightgrey',
    fontsize=12,
    fontweight='bold',
    va='center'
)

# Customize the chart with dark theme
ax.set_title('Edge (Win% - Loss%) by Day of Week', 
             color='white', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', color='white', fontsize=14, fontweight='bold')
ax.set_ylabel('Edge (%)', color='white', fontsize=14, fontweight='bold', rotation=0)
ax.yaxis.set_label_coords(-0.075, 0.6)  # Make the ylabel horizontal

# Style x and y axis
ax.tick_params(axis='x', colors='white')
plt.yticks([])
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Remove borders (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

# Adjust layout
plt.tight_layout()

# Save the chart with dark background
plt.savefig('png/vis/DailyEdge.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='#1E1E1E', 
            edgecolor='none')

plt.close()
