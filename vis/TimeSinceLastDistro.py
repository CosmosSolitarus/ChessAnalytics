import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

# Read the prepared CSV file
df = pd.read_csv("csv/MyGamesPrepared.csv")

# Set dark background style
plt.style.use('seaborn-v0_8')

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set dark background
fig.patch.set_facecolor('#1E1E1E')
ax.set_facecolor('#1E1E1E')

# Prepare the data
time_data = df['TimeSinceLast'].values
log_time = np.log(time_data).reshape(-1, 1)

# Calculate percentages for each time threshold
time_thresholds = [
   (300, "5m"),    # 5 minutes
   (1800, "30m"),  # 30 minutes  
   (7200, "2h"),   # 2 hours
   (28800, "8h"),  # 8 hours
   (57600, "16h"), # 16 hours
   (604800, "7d")  # 7 days
]

percentages = []
for threshold, label in time_thresholds:
   pct = (time_data <= threshold).mean() * 100
   percentages.append(f"<{label} - {pct:.1f}% of games")

# Create text box with percentages
text = '\n'.join(percentages)
props = dict(boxstyle='round', facecolor='#1E1E1E', edgecolor='white', alpha=0.7)
ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', horizontalalignment='right',
       bbox=props, color='white')

# Fit KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(log_time)

# Create a range of points to evaluate the KDE
x_plot = np.linspace(np.min(log_time), np.max(log_time), 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_plot)

# Transform back to original scale for plotting
x_plot_original = np.exp(x_plot)
density = np.exp(log_dens)

# Create the plot
ax.fill_between(x_plot_original.flatten(), density, 
               alpha=0.5, color='#4FB6D6')  # Light blue color
ax.plot(x_plot_original.flatten(), density, 
       color='#4FB6D6', linewidth=2)

# Style the plot
ax.set_xscale('log')
ax.set_title('Distribution of Time Between Games', 
            color='white', pad=50)
ax.set_xlabel('Time Since Last Game (seconds, log scale)', 
            color='white')
ax.set_ylabel('Density', color='white')

# Configure grid
ax.grid(False)

# Style tick labels
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add secondary x-axis with specified time intervals
def seconds_to_time_str(x):
   if x < 3600:
       return f"{int(x/60)}m"
   elif x < 86400:
       return f"{int(x/3600)}h"
   else:
       return f"{int(x/86400)}d"

ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xlim(ax.get_xlim())
time_ticks = [
   300,    # 5 minutes
   1800,   # 30 minutes
   7200,   # 2 hours
   28800,  # 8 hours
   57600,  # 16 hours
   604800  # 168 hours
]
ax2.set_xticks(time_ticks)
ax2.set_xticklabels([seconds_to_time_str(x) for x in time_ticks])
ax2.tick_params(colors='white')
ax2.spines['top'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('png/time_between_games_distribution.png', 
           dpi=300, bbox_inches='tight',
           facecolor='#1E1E1E', edgecolor='none')
plt.close()

print("Time distribution graph has been generated and saved in the 'png' directory.")