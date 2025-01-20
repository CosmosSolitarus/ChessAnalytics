import os
import pandas as pd

def combine_csv(usernames):
    """
    Combines and sorts game data from multiple CSV files for the given usernames.

    Parameters:
        usernames (list of str): List of Chess.com usernames whose game data to combine.
    """
    dataframes = []
    for username in usernames:
        try:
            input_file = f"csv/MyGames{username}.csv"
            df = pd.read_csv(input_file)
            dataframes.append(df)
            print(f"Loaded data for {username} from {input_file}.")
        except Exception as e:
            print(f"Failed to load data for {username} from {input_file}: {e}")
            continue

    if not dataframes:
        print("No dataframes to combine.")
        return

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Convert the 'Date' column to datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

    # Sort the dataframe by 'Date' and 'StartTime'
    sorted_df = combined_df.sort_values(by=['Date', 'StartTime'], na_position='last')

    # Save the sorted dataframe to a new CSV file
    output_file = "csv/MyGamesCombined.csv"
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sorted_df.to_csv(output_file, index=False)
        print(f"Combined and sorted data saved to {output_file}.")
    except Exception as e:
        print(f"Failed to save combined data to {output_file}: {e}")