import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the prepared CSV file
df = pd.read_csv("csv/MyGamesPrepared.csv")

def create_edge_graph(data, games_col, win_perc_col, loss_perc_col, title, filename):
    # Create figure and axis objects with a single subplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    # Calculate mean percentages for each number of games and compute edge
    avg_win = data[data[games_col] > 0].groupby(games_col)[win_perc_col].mean()
    avg_loss = data[data[games_col] > 0].groupby(games_col)[loss_perc_col].mean()
    edge_data = avg_win - avg_loss
    
    # Get sorted unique edges and use min and max for color scaling
    sorted_edges = sorted(edge_data.unique())
    min_edge = sorted_edges[0]
    max_edge = sorted_edges[-1]
    
    # Create color gradient based on edge values
    norm = plt.Normalize(min_edge, max_edge)
    
    # Use RdYlGn colormap (red for negative edge, green for positive edge)
    cmap = plt.cm.RdYlGn
    colors = cmap(norm(edge_data.values))
    
    # Create bar plot with color gradient
    bars = ax.bar(edge_data.index, edge_data.values, color=colors)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    
    # Style settings for dark mode
    ax.set_title(f'{title}\n(Range: {min_edge:.1f}% to {max_edge:.1f}%)', 
                color='white', pad=20)
    ax.set_xlabel('Number of Games Played', color='white')
    ax.set_ylabel('Edge (Win% - Loss%)', color='white')
    
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
    cbar = fig.colorbar(sm, ax=ax, label=f'Edge (Range: {min_edge:.1f}% to {max_edge:.1f}%)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'png/{filename}.png', dpi=300, bbox_inches='tight',
                facecolor='#1E1E1E', edgecolor='none')
    plt.close()

# Create edge graphs
create_edge_graph(df, 'GameOfDay', 'DailyWinPerc', 'DailyLossPerc',
                 'Win-Loss Edge vs Games Played per Day', 'daily_edge')
create_edge_graph(df, 'GameOfWeek', 'WeeklyWinPerc', 'WeeklyLossPerc',
                 'Win-Loss Edge vs Games Played per Week', 'weekly_edge')

print("Edge graphs have been generated and saved in the 'png' directory.")