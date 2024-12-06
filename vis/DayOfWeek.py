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

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
daily_percentages.plot(kind='bar', stacked=True, color=['#137d3f', '#d62728', '#1f77b4'], ax=plt.gca())

# Customize the chart
plt.title('Win, Loss, and Draw Percentages by Day of Week (10 Minute Games)', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold', rotation=45)
plt.yticks(fontsize=12, fontweight='bold')

# Customize legend
plt.legend(
    title='Result',
    fontsize=14,        # Larger text
    title_fontsize=16,  # Larger and bold title
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,      # Remove the box around the legend
    prop={'weight': 'bold'}  # Make legend text bold
)

# Remove borders (spines)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Adjust layout to add extra space on the right
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave extra space on the right for the legend

# Display the chart
plt.show()
