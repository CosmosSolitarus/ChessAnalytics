import pandas as pd
import matplotlib.pyplot as plt

# Read the prepared CSV file
df = pd.read_csv("csv/MyGamesPrepared.csv")

def calculate_frequency(data, column):
    """Calculate percentage frequency of each game number"""
    freq = data[column].value_counts()
    freq_percent = (freq / len(data) * 100).sort_index()
    return freq_percent

def create_frequency_graph(data, column, title, filename):
    # Create figure and axis objects with a single subplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    # Calculate frequency percentages
    freq_percent = calculate_frequency(data, column)
    
    # Get sorted unique percentages and use min and max for color scaling
    sorted_percs = sorted(freq_percent.unique())
    min_perc = sorted_percs[0]
    max_perc = sorted_percs[-1]
    
    # Create color gradient
    norm = plt.Normalize(min_perc, max_perc)
    cmap = plt.cm.viridis  # Using viridis for frequency visualization
    colors = cmap(norm(freq_percent.values))
    
    # Create bar plot with color gradient
    bars = ax.bar(freq_percent.index, freq_percent.values, color=colors)
    
    # Style settings for dark mode
    ax.set_title(f'{title}\n(Range: {min_perc:.1f}% - {max_perc:.1f}%)', 
                color='white', pad=20)
    ax.set_xlabel('Game Number', color='white')
    ax.set_ylabel('Frequency (%)', color='white')
    
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
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=f'Frequency (Range: {min_perc:.1f}% - {max_perc:.1f}%)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'png/{filename}.png', dpi=300, bbox_inches='tight',
                facecolor='#1E1E1E', edgecolor='none')
    plt.close()

# Create frequency graphs
create_frequency_graph(df, 'GameOfDay', 
                      'Frequency Distribution of Games Played per Day',
                      'daily_game_frequency')
create_frequency_graph(df, 'GameOfWeek',
                      'Frequency Distribution of Games Played per Week',
                      'weekly_game_frequency')

print("Frequency graphs have been generated and saved in the 'png' directory.")