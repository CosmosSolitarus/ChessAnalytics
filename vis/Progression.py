import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def Progression():
    # Read the CSV file
    df = pd.read_csv('csv/MyGamesCombined.csv')

    # Convert the "Date" column to datetime format for filtering
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the data for "Cosmos_IV", TimeControl == 600 (10 minutes), after May 31, 2024, and above 1200 Elo
    cosmos_iv_data = df[
        (df['Account'] == 'Cosmos_IV') & 
        (df['TimeControl'] == 600) & 
        (df['Date'] >= '2024-05-31') &
        (df['MyElo'] >= 1200)
    ]

    # Initialize variables for tracking ATH and color values
    ath = -float('inf')  # Start with a very low value
    colors = []  # Store colors for each point

    # Iterate through "GameNumber" in ascending order and determine colors
    for elo in cosmos_iv_data['MyElo']:
        if elo > ath:
            ath = elo  # Update ATH
            colors.append('#FFD700')  # Color ATH as gold
        else:
            dec_max = 100/7
            
            percentage_decrease = (ath - elo) / ath * 100  # Calculate % decrease from ATH
            percentage_decrease = min(percentage_decrease, dec_max)

            # Normalize percentage decrease for color gradient
            norm = mcolors.Normalize(vmin=0, vmax=dec_max)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "ath_gradient", ['#0066FF', '#FF3333']
            )
            colors.append(cmap(norm(percentage_decrease)))  # Map percentage to gradient color

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(30, 8))

    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')

    # Scatter plot Elo progression with custom colors
    ax.scatter(
        cosmos_iv_data['GameNumber'], 
        cosmos_iv_data['MyElo'], 
        c=colors, 
        s=50,
        edgecolor='none'
    )

    # Customize the chart
    ax.set_title('Elo Progression (10 Minute Games, 5/31/24 - Present)', 
                 color='white', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Game Number', color='white', fontsize=18, fontweight='bold', labelpad=30)
    ax.set_ylabel('Elo', color='white', fontsize=18, fontweight='bold', rotation=0, labelpad=30)

    # Rotate the "Elo" label sideways
    ax.yaxis.set_label_coords(-0.05, 0.5)

    # Style x and y axis
    ax.tick_params(axis='x', colors='white', length=0, labelsize=16)
    ax.tick_params(axis='y', colors='white', length=0, labelsize=16)

    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#3A3A3A', alpha=0.5)

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
    cbar.set_label('% Decrease \nfrom Previous \nBest', color='white', fontsize=18, fontweight='bold', rotation=0, labelpad=60)
    cbar.ax.yaxis.set_tick_params(color='white', length=0, labelsize=16)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Show only min and max labels on color gradient
    cbar.set_ticks([0, dec_max])
    cbar.set_ticklabels(['0', f'{dec_max:.1f}'])

    # Adjust layout and save the chart with dark background
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('png/vis/Progression.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='#1E1E1E', 
                edgecolor='none')
    
    plt.close()

if __name__ == "__main__":
    Progression()
