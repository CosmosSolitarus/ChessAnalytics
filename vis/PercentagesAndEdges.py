import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_analysis_table(csv_file='csv/MyGamesCombined.csv'):
    """
    Creates a table analyzing chess game results across different eras and colors.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing chess game data.
    
    Returns:
    --------
    None, saves the table as an image file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert the "Date" column to datetime format for filtering
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter to only include games by Cosmos_IV with 600 time control
    df = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]
    
    # Define reference games for each era
    # Default era - games from and after the one with Elo 803 on May 31, 2024
    default_reference_start = df[
        (df['Date'] >= '2024-05-31') & 
        (df['MyElo'] == 803)
    ].iloc[0]['GameNumber']
    
    # Historical era - games between Elo 803 (May 31, 2024) and Elo 1800 (August 27, 2024)
    historical_reference_start = default_reference_start
    historical_reference_end = df[
        (df['Date'] >= '2024-08-27') & 
        (df['MyElo'] == 1800)
    ].iloc[0]['GameNumber']
    
    # Baseline era - games between Elo 1582 (July 1, 2024) and Elo 1800 (August 27, 2024)
    baseline_reference_start = df[
        (df['Date'] >= '2024-07-01') & 
        (df['MyElo'] == 1582)
    ].iloc[0]['GameNumber']
    baseline_reference_end = historical_reference_end
    
    # Dark era - games between losing game with Elo 1792 (August 27, 2024) and Elo 1575 (November 3, 2024)
    dark_reference_start = df[
        (df['Date'] >= '2024-08-27') & 
        (df['MyElo'] == 1792) & 
        (df['Result'] == 'lost')
    ].iloc[0]['GameNumber']
    
    # Modern era - games from and after the one with Elo 1575 on Nov 3, 2024
    modern_reference = df[
        (df['Date'] >= '2024-11-03') & 
        (df['MyElo'] == 1575)
    ].iloc[0]['GameNumber']
    
    dark_reference_end = modern_reference - 1
    
    # Recent era - games from and after the one with Elo 1807 on Jan 24, 2025
    recent_reference = df[
        (df['Date'] >= '2025-01-24') & 
        (df['MyElo'] == 1807)
    ].iloc[0]['GameNumber']
    
    # Filter data for each era
    default_data = df[df['GameNumber'] >= default_reference_start]
    historical_data = df[(df['GameNumber'] >= historical_reference_start) & (df['GameNumber'] <= historical_reference_end)]
    baseline_data = df[(df['GameNumber'] >= baseline_reference_start) & (df['GameNumber'] <= baseline_reference_end)]
    dark_data = df[(df['GameNumber'] >= dark_reference_start) & (df['GameNumber'] <= dark_reference_end)]
    modern_data = df[df['GameNumber'] >= modern_reference]
    recent_data = df[df['GameNumber'] >= recent_reference]
    
    # Initialize results dictionary
    results = {}
    
    # Calculate statistics for each category (in the desired order)
    # Default era
    results['All games'] = calculate_stats(default_data)
    results['All games as white'] = calculate_stats(default_data[default_data['Color'] == 'white'])
    results['All games as black'] = calculate_stats(default_data[default_data['Color'] == 'black'])
    
    # Historical era
    results['All games in the historical era'] = calculate_stats(historical_data)
    results['All games as white in the historical era'] = calculate_stats(historical_data[historical_data['Color'] == 'white'])
    results['All games as black in the historical era'] = calculate_stats(historical_data[historical_data['Color'] == 'black'])
    
    # Baseline era
    results['All games in the baseline era'] = calculate_stats(baseline_data)
    results['All games as white in the baseline era'] = calculate_stats(baseline_data[baseline_data['Color'] == 'white'])
    results['All games as black in the baseline era'] = calculate_stats(baseline_data[baseline_data['Color'] == 'black'])
    
    # Dark era
    results['All games in the dark era'] = calculate_stats(dark_data)
    results['All games as white in the dark era'] = calculate_stats(dark_data[dark_data['Color'] == 'white'])
    results['All games as black in the dark era'] = calculate_stats(dark_data[dark_data['Color'] == 'black'])
    
    # Modern era
    results['All games in the modern era'] = calculate_stats(modern_data)
    results['All games as white in the modern era'] = calculate_stats(modern_data[modern_data['Color'] == 'white'])
    results['All games as black in the modern era'] = calculate_stats(modern_data[modern_data['Color'] == 'black'])
    
    # Recent era
    results['All games in the recent era'] = calculate_stats(recent_data)
    results['All games as white in the recent era'] = calculate_stats(recent_data[recent_data['Color'] == 'white'])
    results['All games as black in the recent era'] = calculate_stats(recent_data[recent_data['Color'] == 'black'])
    
    # Create a table and save as an image
    create_table_image(results, 'png/vis/Table.png')
    
    return results

def calculate_stats(data):
    """
    Calculate statistics for a subset of games.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the games to analyze.
    
    Returns:
    --------
    dict : Dictionary with calculated statistics.
    """
    total_games = len(data)
    
    if total_games == 0:
        return {
            'Wins': 0,
            'Draws': 0,
            'Losses': 0,
            'Win Percentage': 0.00,
            'Draw Percentage': 0.00,
            'Loss Percentage': 0.00,
            'Edge': 0,
            'Edge Percentage': 0.00
        }
    
    # Count wins, draws, and losses
    wins = len(data[data['Result'] == 'won'])
    draws = len(data[data['Result'] == 'draw'])
    losses = len(data[data['Result'] == 'lost'])
    
    # Calculate percentages
    win_percentage = (wins / total_games) * 100
    draw_percentage = (draws / total_games) * 100
    loss_percentage = (losses / total_games) * 100
    
    # Calculate edge metrics
    edge = wins - losses
    edge_percentage = win_percentage - loss_percentage

    return {
        'Wins': wins,
        'Draws': draws,
        'Losses': losses,
        'Win Percentage': round(win_percentage, 2),
        'Draw Percentage': round(draw_percentage, 2),
        'Loss Percentage': round(loss_percentage, 2),
        'Edge': edge,
        'Edge Percentage': round(edge_percentage, 2)
    }

def create_table_image(results, output_file):
    """
    Creates a table image from the calculated statistics.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the calculated statistics.
    output_file : str
        Path to save the table image.
    
    Returns:
    --------
    None, saves the image to the specified path.
    """
    # Define row labels and column headers
    row_labels = list(results.keys())
    column_headers = ['Wins', 'Draws', 'Losses', 'Win %', 'Draw %', 'Loss %', 'Edge', 'Edge %']
    
    # Extract data for the table
    data = []
    for label in row_labels:
        row_data = []
        stats = results[label]
        row_data.append(stats['Wins'])
        row_data.append(stats['Draws'])
        row_data.append(stats['Losses'])
        row_data.append(f"{stats['Win Percentage']:.2f}%")
        row_data.append(f"{stats['Draw Percentage']:.2f}%")
        row_data.append(f"{stats['Loss Percentage']:.2f}%")
        row_data.append(stats['Edge'])
        row_data.append(f"{stats['Edge Percentage']:.2f}%")
        data.append(row_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    # Turn off axes
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=data,
        rowLabels=row_labels,
        colLabels=column_headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Adjust table styling for dark theme
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(color='white')
        
        # Header row
        if i == 0:
            cell.set_facecolor('#0066cc')
            cell.set_text_props(weight='bold')
        # Row labels column
        elif j == -1:
            cell.set_facecolor('#004080')
            cell.set_text_props(weight='bold')
        # Data cells - alternate colors for better readability
        else:
            if i % 2 == 0:
                cell.set_facecolor('#333333')
            else:
                cell.set_facecolor('#444444')
    
    # Add a title
    plt.suptitle('Chess Games Analysis - Cosmos_IV (10 Minute Games)', 
                 color='white', fontsize=18, fontweight='bold', y=0.98)
    
    # Save the table with dark background
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='#1E1E1E', 
                edgecolor='none')
    
    plt.close()

if __name__ == "__main__":
    create_analysis_table()