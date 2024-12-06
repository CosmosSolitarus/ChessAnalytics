import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('csv/MyGamesCombined.csv')

# Filter the data for "Cosmos_IV" with TimeControl == 600
cosmos_iv_data = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]

# Plot the line chart with a thicker line
plt.figure(figsize=(10, 6))
plt.plot(cosmos_iv_data['Date'], cosmos_iv_data['MyElo'], color='#137d3f', linewidth=4)

# Customize the chart
plt.title('My Elo Progression 2020 to Present (10 Minute Games)', fontsize=24, fontweight='bold')
plt.xlabel('Date', fontsize=20, fontweight='bold')
plt.ylabel('Elo', fontsize=20, fontweight='bold')

# Format x-axis to show only the year, and remove x-axis ticks and labels
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.xticks([])  # Remove x-axis ticks
#plt.gca().get_xaxis().set_ticks_position('none')  # Remove x-axis line

# Remove the border (spines)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Increase the font size for ticks (if any visible)
plt.tick_params(axis='both', which='major', labelsize=12, width=1.5)

# Display the chart
plt.tight_layout()
plt.show()
