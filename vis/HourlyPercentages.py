import pandas as pd
import matplotlib.pyplot as plt

def HourlyPercentages():
    # Read the CSV file
    df = pd.read_csv('csv/MyGamesCombined.csv')

    # Filter data for "Cosmos_IV" with TimeControl = 600
    filtered_data = df[(df['Account'] == 'Cosmos_IV') & (df['TimeControl'] == 600)]

    # Group data by "HourOfDay" and "Result", then calculate the counts
    hourly_results = filtered_data.groupby(['HourOfDay', 'Result']).size().unstack(fill_value=0)

    # Calculate percentages for each "HourOfDay"
    hourly_percentages = hourly_results.div(hourly_results.sum(axis=1), axis=0) * 100

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set dark background
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')

    # Create stacked bar chart
    bars = hourly_percentages.plot(kind='bar', stacked=True, color=['#137d3f', '#d62728', '#1f77b4'], ax=ax)

    # Add data labels
    for c in bars.containers:
        bars.bar_label(c, label_type='center', color='white', fontweight='bold', fmt='%.1f')

    # Customize the chart with dark theme
    ax.set_title('Win, Loss, and Draw Percentages by Hour of Day (10 Minute Games)', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', color='white', fontsize=14, fontweight='bold')

    # Style x and y axis
    ax.tick_params(axis='x', colors='white', rotation=0)
    ax.tick_params(axis='y', colors='white')

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        reversed(handles),
        reversed(labels),
        title='Result',
        title_fontsize=14,
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        prop={'weight': 'bold'},
        labelcolor='white'
    )
    plt.setp(legend.get_title(), color='white')

    # Remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adjust layout to add extra space for the legend
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Save the chart with dark background
    plt.savefig('png/vis/HourlyPercentages.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='#1E1E1E', 
                edgecolor='none')

    # Close the plot to free up memory
    plt.close()

if __name__ == "__main__":
    HourlyPercentages()