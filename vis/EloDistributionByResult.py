import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def EloDistributionByResult():
    # Read the CSV file
    df = pd.read_csv('csv/MyGamesCombined.csv')

    # Filter data for "Cosmos_IV" with TimeControl = 600
    filtered_data = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]

    # Calculate Elo Difference
    filtered_data['EloDiff'] = filtered_data['MyElo'] - filtered_data['OppElo']

    # Separate data by result
    wins = filtered_data[filtered_data['Result'] == 'won']
    losses = filtered_data[filtered_data['Result'] == 'lost']
    draws = filtered_data[filtered_data['Result'] == 'draw']

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')

    # Plot overlapping KDEs
    sns.kdeplot(wins['EloDiff'], fill=True, alpha=0.6, color='#137d3f', label='Wins', ax=ax)
    sns.kdeplot(losses['EloDiff'], fill=True, alpha=0.6, color='#d62728', label='Losses', ax=ax)
    sns.kdeplot(draws['EloDiff'], fill=True, alpha=0.6, color='#1f77b4', label='Draws', ax=ax)

    # Restrict the x-axis
    ax.set_xlim(-100, 100)

    # Customize the chart with dark theme
    ax.set_title('Distribution of Elo Difference by Result', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Elo Difference (MyElo - OppElo)', 
                  color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', 
                  color='white', fontsize=14, fontweight='bold')

    # Style x and y axis
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Customize legend
    legend = ax.legend(
        title='Result',
        fontsize=14,
        title_fontsize=16,
        frameon=False,
        prop={'weight': 'bold'},
        labelcolor='white'
    )
    plt.setp(legend.get_title(), color='white')

    # Remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the chart with dark background
    plt.savefig('png/vis/EloDistributionByResult.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='#1E1E1E', 
                edgecolor='none')

    # Close the plot to free up memory
    plt.close()

if __name__ == "__main__":
    EloDistributionByResult()