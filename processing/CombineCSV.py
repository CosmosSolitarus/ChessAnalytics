import pandas as pd

# Load the CSV files into dataframes
df1 = pd.read_csv('csv/MyGamesCosmosSolitarus.csv')
df2 = pd.read_csv('csv/MyGamesCosmos_IV.csv')

# Combine the two dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

# Convert the 'Date' column to datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Sort the dataframe by 'Date' first, and then by 'StartTime' (since StartTime is in HH:MM:SS format, no need for conversion)
sorted_df = combined_df.sort_values(by=['Date', 'StartTime'])

# Save the sorted dataframe to a new CSV file called 'MyGamesCombined.csv'
sorted_df.to_csv('csv/MyGamesCombined.csv', index=False)

# Print the sorted dataframe (optional)
print(sorted_df)
