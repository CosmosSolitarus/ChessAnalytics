import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def create_elo_progression_chart(csv_file='csv/MyGamesCombined.csv', mode='default'):
    """
    Creates a scatter plot of chess Elo progression over time.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing chess game data.
    mode : str
        'default': Shows all 10-minute games from May 31, 2024 onward with 15% max decrease.
        'modern': Shows games from and after the one with Elo 1575 on Nov 3, 2024 with 5% max decrease.
        'recent': Shows games from and after the one with Elo 1807 on Jan 24, 2025 with 2% max decrease.
    
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        The figure and axis objects created for the plot.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert the "Date" column to datetime format for filtering
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set parameters based on mode
    if mode == 'default':
        # Default mode: games from and after the one with Elo 803 on May 31, 2024
        # Find the specific reference game
        reference_game = df[
            (df['Account'] == 'Cosmos_IV') & 
            (df['TimeControl'] == 600) & 
            (df['Date'] >= '2024-05-31') & 
            (df['MyElo'] == 803)
        ]
        
        reference_game_number = reference_game.iloc[0]['GameNumber']
        
        # Filter games including and after this reference game
        filtered_data = df[
            (df['Account'] == 'Cosmos_IV') & 
            (df['TimeControl'] == 600) & 
            (df['GameNumber'] >= reference_game_number)
        ]
        
        max_decrease = 15
        output_file = 'png/vis/Progression.png'
    elif mode == 'modern':
        # Modern mode: games from and after the one with Elo 1575 on Nov 3, 2024
        # Find the specific reference game
        reference_game = df[
            (df['Account'] == 'Cosmos_IV') & 
            (df['TimeControl'] == 600) & 
            (df['Date'] >= '2024-11-03') & 
            (df['MyElo'] == 1575)
        ]
        
        reference_game_number = reference_game.iloc[0]['GameNumber']
        
        # Filter games including and after this reference game
        filtered_data = df[
            (df['Account'] == 'Cosmos_IV') & 
            (df['TimeControl'] == 600) & 
            (df['GameNumber'] >= reference_game_number)
        ]
        
        max_decrease = 5
        output_file = 'png/vis/ModernProgression.png'
    elif mode == 'recent':
        # Recent mode: games from and after the one with Elo 1807 on Jan 24, 2025
        # Find the specific reference game
        reference_game = df[
            (df['Account'] == 'Cosmos_IV') & 
            (df['TimeControl'] == 600) & 
            (df['Date'] >= '2025-01-24') & 
            (df['MyElo'] == 1807)
        ]
        
        reference_game_number = reference_game.iloc[0]['GameNumber']
        
        # Filter games including and after this reference game
        filtered_data = df[
            (df['Account'] == 'Cosmos_IV') & 
            (df['TimeControl'] == 600) & 
            (df['GameNumber'] >= reference_game_number)
        ]
        
        max_decrease = 2
        output_file = 'png/vis/RecentProgression.png'
    else:
        raise ValueError("Mode must be 'default', 'modern', or 'recent'")
    
    # Initialize variables for tracking ATH and color values
    ath = -float('inf')  # Start with a very low value
    colors = []  # Store colors for each point
    
    # Create color map for percentage decrease (define it here, outside the loop)
    norm = mcolors.Normalize(vmin=0, vmax=max_decrease)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ath_gradient", ['#0066FF', '#FF3333']
    )
    
    # Iterate through "GameNumber" in ascending order and determine colors
    for elo in filtered_data['MyElo']:
        if elo > ath:
            ath = elo  # Update ATH
            colors.append('#FFD700')  # Color ATH as gold
        else:
            percentage_decrease = (ath - elo) / ath * 100  # Calculate % decrease from ATH
            percentage_decrease = min(percentage_decrease, max_decrease)  # Cap percentage decrease
            
            # Use the already defined norm and cmap to get color
            colors.append(cmap(norm(percentage_decrease)))  # Map percentage to gradient color
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(30, 8))
    
    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    # Create a new column for renumbered game sequence
    # This creates a sequential numbering starting from 1
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data['GameSequence'] = filtered_data.index + 1
    
    # Scatter plot Elo progression with custom colors and renumbered game sequence
    ax.scatter(
        filtered_data['GameSequence'], 
        filtered_data['MyElo'], 
        c=colors, 
        s=50,  # Size of points
        edgecolor='none'
    )
    
    # Customize the chart
    if mode == "default":
        mode_label = "Default"
    elif mode == "modern":
        mode_label = "Modern"
    elif mode == "recent":
        mode_label = "Recent"
    ax.set_title(f'Elo Progression - {mode_label} (10 Minute Games)', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Game Sequence', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Elo', color='white', fontsize=14, fontweight='bold')
    
    # Style x and y axis
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add a colorbar to show the percentage decrease
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
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='#1E1E1E', 
                edgecolor='none')
    
    plt.close()

    return fig

def all_charts():
    create_elo_progression_chart(mode='default')
    create_elo_progression_chart(mode='modern')
    create_elo_progression_chart(mode='recent')

if __name__ == "__main__":
    all_charts()