import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('csv/MyGamesCombined.csv')

# Filter data for "Cosmos_IV" with TimeControl = 600
filtered_data = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]

# Group data by "DayOfWeek" and "Result", then calculate the counts
daily_results = filtered_data.groupby(['DayOfWeek', 'Result']).size().unstack(fill_value=0)

# Calculate percentages for each "DayOfWeek"
daily_percentages = daily_results.div(daily_results.sum(axis=1), axis=0) * 100

# Rename index to first letters of days (1-indexed)
day_map = {1: 'M', 2: 'T', 3: 'W', 4: 'T', 5: 'F', 6: 'S', 7: 'S'}
daily_percentages.index = [day_map[idx] for idx in daily_percentages.index]

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(12, 8))

# Set dark background
fig.patch.set_facecolor('#1E1E1E')
ax.set_facecolor('#1E1E1E')

# Reorder results and rename
daily_percentages = daily_percentages.rename(columns={
    'draw': 'Draw', 
    'lost': 'Loss', 
    'won': 'Win'
})

# Create stacked bar chart
bars = daily_percentages.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728', '#137d3f'], ax=ax)

# Add data labels
for c in bars.containers:
    bars.bar_label(c, label_type='center', color='white', fontweight='bold', fmt='%.1f')

# Customize the chart with dark theme
ax.set_title('Win, Loss, and Draw Percentages by Day of Week (10 Minute Games)', 
             color='white', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', color='white', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)', color='white', fontsize=14, fontweight='bold')

# Style x and y axis
ax.tick_params(axis='x', colors='white', rotation=45)
ax.tick_params(axis='y', colors='white')

# Customize legend
handles, labels = ax.get_legend_handles_labels()

legend = ax.legend(
    reversed(handles),
    reversed(labels),
    title='Result',
    title_fontsize=16,
    fontsize=14,
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    prop={'weight': 'bold'},
    labelcolor='white'
)

plt.setp(legend.get_title(), color='white')

# Remove borders (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.95, 1])

# Save the chart with dark background
plt.savefig('png/vis/DailyPercentages.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='#1E1E1E', 
            edgecolor='none')

plt.close()