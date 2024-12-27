import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the prepared CSV file
df = pd.read_csv("csv/MyGamesPrepared.csv")

# Set dark background style
plt.style.use('seaborn-v0_8')

def create_graph(data, games_col, perc_col, title, filename):
    # Create figure and axis objects with a single subplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')  # Dark background for figure
    ax.set_facecolor('#1E1E1E')  # Dark background for plot area
    
    # Calculate mean percentage for each number of games
    # Exclude 0 games as it shouldn't exist
    avg_data = data[data[games_col] > 0].groupby(games_col)[perc_col].mean()
    
    # Get sorted unique percentages and use 2nd lowest and highest
    sorted_percs = sorted(avg_data.unique())
    min_perc = sorted_percs[1] if len(sorted_percs) > 1 else sorted_percs[0]
    max_perc = sorted_percs[-1]
    
    # Create color gradient based on percentage values, using data-specific range
    norm = plt.Normalize(min_perc, max_perc)
    
    # Choose appropriate colormap based on the type of percentage
    if 'Win' in title:
        cmap = plt.cm.Greens
    elif 'Draw' in title:
        cmap = plt.cm.Blues
    else:  # Loss
        cmap = plt.cm.Reds
    
    colors = cmap(norm(avg_data.values))
    
    # Create bar plot with color gradient
    bars = ax.bar(avg_data.index, avg_data.values, color=colors)
    
    # Style settings for dark mode
    ax.set_title(f'{title}\n(Range: {min_perc:.1f}% - {max_perc:.1f}%)', color='white', pad=20)
    ax.set_xlabel('Number of Games Played', color='white')
    ax.set_ylabel('Percentage', color='white')
    
    # Configure grid - only vertical lines
    ax.grid(True, axis='y', alpha=0.2, color='white')
    ax.grid(False, axis='x')
    
    # Style the axis lines and labels
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    
    # Style tick labels
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Add colorbar with data-specific range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=f'Percentage (Range: {min_perc:.1f}% - {max_perc:.1f}%)')
    cbar.ax.yaxis.label.set_color('white')  # Colorbar label color
    cbar.ax.tick_params(colors='white')  # Colorbar tick color
    
    # Adjust layout to prevent colorbar overlap
    plt.tight_layout()
    
    # Save the plot with dark background
    plt.savefig(f'png/{filename}.png', dpi=300, bbox_inches='tight', 
                facecolor='#1E1E1E', edgecolor='none')
    plt.close()

# Create daily stats graphs
create_graph(df, 'GameOfDay', 'DailyWinPerc', 
            'Win Percentage vs Games Played (Daily)', 'daily_wins')
create_graph(df, 'GameOfDay', 'DailyDrawPerc', 
            'Draw Percentage vs Games Played (Daily)', 'daily_draws')
create_graph(df, 'GameOfDay', 'DailyLossPerc', 
            'Loss Percentage vs Games Played (Daily)', 'daily_losses')

# Create weekly stats graphs
create_graph(df, 'GameOfWeek', 'WeeklyWinPerc', 
            'Win Percentage vs Games Played (Weekly)', 'weekly_wins')
create_graph(df, 'GameOfWeek', 'WeeklyDrawPerc', 
            'Draw Percentage vs Games Played (Weekly)', 'weekly_draws')
create_graph(df, 'GameOfWeek', 'WeeklyLossPerc', 
            'Loss Percentage vs Games Played (Weekly)', 'weekly_losses')

print("Graphs have been generated and saved in the 'png' directory.")