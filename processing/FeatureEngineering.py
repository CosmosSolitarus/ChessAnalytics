import pandas as pd

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
    'TimeSinceLast'
]
df = df[columns_to_keep]

# Drop rows with any NA values
df = df.dropna()

# Save the prepared CSV
df.to_csv("csv/MyGamesPrepared.csv", index=False)

print("CSV preparation complete! The file has been saved as 'MyGamesPrepared.csv'.")
