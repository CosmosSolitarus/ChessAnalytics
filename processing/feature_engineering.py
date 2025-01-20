import pandas as pd

def feature_engineering():
    # Load the CSV file
    df = pd.read_csv("csv/MyGamesCombined.csv")

    # Prepare the data for the new CSV
    # 1. Account - 0 if "Account" is "Cosmos_IV", else 1
    df['Account'] = df['Account'].apply(lambda x: 0 if x == "Cosmos_IV" else 1)

    # 2. One-hot encoding for days of the week based on "Date"
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        df[f'Is{day}'] = (df['Date'].dt.day_name() == day).astype(int)

    # 3. TimeOfDay - Number of seconds that have occurred in the day
    df['TimeOfDay'] = df['StartTime'].apply(lambda x: sum(int(t) * sec for t, sec in zip(x.split(':'), [3600, 60, 1])))

    # 4. EloDifference - MyElo minus OppElo
    df['EloDifference'] = df['MyElo'] - df['OppElo']

    # 5. Color - 0 if "Color" is "white", else 1
    df['Color'] = df['Color'].apply(lambda x: 0 if x == "white" else 1)

    # 6. Result - 0 if "won", 1 if "draw", 2 if "lost"
    result_mapping = {'won': 0, 'draw': 1, 'lost': 2}
    df['Result'] = df['Result'].map(result_mapping)

    # 7. Castle columns - 0 if "No", else 1
    castle_columns = ['ICastledFirst', 'ICastledShort', 'ICastledLong', 'OppCastledShort', 'OppCastledLong']
    for col in castle_columns:
        df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)

    # 8. One-hot encoding for LastResult
    df['LastResultIsWin'] = (df['LastResult'] == 'won').astype(int)
    df['LastResultIsDraw'] = (df['LastResult'] == 'draw').astype(int)
    df['LastResultIsLoss'] = (df['LastResult'] == 'lost').astype(int)

    # 9. One-hot encoding for 2ndLastResult
    df['2ndLastResultIsWin'] = (df['2ndLastResult'] == 'won').astype(int)
    df['2ndLastResultIsDraw'] = (df['2ndLastResult'] == 'draw').astype(int)
    df['2ndLastResultIsLoss'] = (df['2ndLastResult'] == 'lost').astype(int)

    # 10. One-hot encoding for ECO values
    valid_ecos = ['A00', 'A40', 'A45', 'B10', 'B12', 'B13', 'D00', 'D02', 'D10']
    # Create ECO columns initialized to 0
    for eco in valid_ecos:
        df[f'ECO_{eco}'] = 0
    df['ECO_Other'] = 0

    # Set appropriate ECO column to 1 based on the ECO value
    for idx, row in df.iterrows():
        eco = row['ECO']
        if eco in valid_ecos:
            df.at[idx, f'ECO_{eco}'] = 1
        else:
            df.at[idx, 'ECO_Other'] = 1

    # Function to calculate percentages for each row
    def calculate_percentages(df, current_index):
        """
        Calculate win/draw/loss percentages for daily and weekly games up to current_index
        """
        current_row = df.iloc[current_index]
        god = current_row['GameOfDay']
        gow = current_row['GameOfWeek']
        
        # Initialize percentages
        daily_stats = {'win': 0, 'loss': 0, 'draw': 0}
        weekly_stats = {'win': 0, 'loss': 0, 'draw': 0}
        
        # If this is not the first game, calculate percentages
        if current_index > 0:
            # Get previous games in the day
            daily_window = min(int(god) - 1, current_index)
            if daily_window > 0:
                daily_games = df.iloc[current_index - daily_window:current_index]
                daily_results = daily_games['Result'].value_counts(normalize=True)
                daily_stats['win'] = daily_results.get(0, 0) * 100
                daily_stats['loss'] = daily_results.get(2, 0) * 100
                daily_stats['draw'] = daily_results.get(1, 0) * 100
            
            # Get previous games in the week
            weekly_window = min(int(gow) - 1, current_index)
            if weekly_window > 0:
                weekly_games = df.iloc[current_index - weekly_window:current_index]
                weekly_results = weekly_games['Result'].value_counts(normalize=True)
                weekly_stats['win'] = weekly_results.get(0, 0) * 100
                weekly_stats['loss'] = weekly_results.get(2, 0) * 100
                weekly_stats['draw'] = weekly_results.get(1, 0) * 100
        
        return pd.Series({
            'DailyWinPerc': daily_stats['win'],
            'DailyLossPerc': daily_stats['loss'],
            'DailyDrawPerc': daily_stats['draw'],
            'WeeklyWinPerc': weekly_stats['win'],
            'WeeklyLossPerc': weekly_stats['loss'],
            'WeeklyDrawPerc': weekly_stats['draw']
        })

    # Calculate percentages for each row
    percentage_columns = pd.DataFrame([
        calculate_percentages(df, i) for i in range(len(df))
    ])

    # Add new columns to the dataframe
    df = pd.concat([df, percentage_columns], axis=1)

    # Round all percentage columns to 2 decimal places
    percentage_cols = [
        'DailyWinPerc', 'DailyLossPerc', 'DailyDrawPerc',
        'WeeklyWinPerc', 'WeeklyLossPerc', 'WeeklyDrawPerc'
    ]
    df[percentage_cols] = df[percentage_cols].round(2)

    # Select only the desired columns
    columns_to_keep = [
        'Account',
        'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday',
        'TimeOfDay',
        'GameOfDay', 'GameOfWeek', 'TimeControl',
        'EloDifference',
        'Color',
        'Result',
        'ICastledFirst', 'ICastledShort', 'ICastledLong', 'OppCastledShort', 'OppCastledLong',
        'LastResultIsWin', 'LastResultIsDraw', 'LastResultIsLoss',
        '2ndLastResultIsWin', '2ndLastResultIsDraw', '2ndLastResultIsLoss',
        'MyNumMoves', 'OppNumMoves',
        'MyTotalTime', 'OppTotalTime',
        'MyAvgTPM', 'OppAvgTPM',
        'TimeSinceLast',
        'DailyWinPerc', 'DailyLossPerc', 'DailyDrawPerc',
        'WeeklyWinPerc', 'WeeklyLossPerc', 'WeeklyDrawPerc',
        'ECO_A00', 'ECO_A40', 'ECO_A45', 'ECO_B10', 'ECO_B12', 'ECO_B13', 
        'ECO_D00', 'ECO_D02', 'ECO_D10', 'ECO_Other'
    ]

    df = df[columns_to_keep]

    #df = df[df['TimeControl'] == 600]

    # Drop rows with any NA values
    df = df.dropna()
    df = df.iloc[1:]

    # Save the prepared CSV
    df.to_csv("csv/MyGamesPrepared.csv", index=False)