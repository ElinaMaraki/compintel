import pandas as pd

df = pd.read_csv('pokemon_data.csv')
# pd.read_excel('---.xlsx') for excel files
# pd.read_csv('pokemon_data.txt', delimiter='\t') tab seperated file

# print(df)
# print(df.head(3))  # default 5
# print(df.tail())

#  Read Headers
# print(df.columns)

# Read each column w/ column name
# print(df['Name'])  # df.Name but doesn't work for 2 word column name
# print(df['Name'][0:5])  # top 5 names
# print(df[['Name', 'Type', 'HP']])  # more than 1 columns

# print row 1
# print(df.iloc[1])
# print rows 0 to 4
# print(df.iloc[0:4])

# Get specific Location [Row, Column]
# print(df.iloc[2, 1])

# Locate based on conditional standards
# print(df.loc[df['Type 1'] == "Fire"])

# Iterate through rows
# for index, row in df.iterrows():
#     print(index, row)


# metrics of data
# print(df.describe())

# Sort data bases on column value
# print(df.sort_values('Name', ascending=False))
# print(df.sort_values(['Type 1', 'HP'], ascending=[True, False]))

# Create new column in data frame
# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']

# Create column the horizontal sum of all rows of columns 4 to 9
df['Total'] = df.iloc[:, 4:10].sum(axis=1)
# df['Total'] = df.iloc[:, ['HP']:['Speed']].sum(axis=1)

# Choose where to put your column
# cols = list(df.columns)
# df = df[cols[0:4] + [cols[-1]] + cols[4:12]]

# Drop columns
# df = df.drop(columns='Total')
print(df.head(5))

# Save dataframe to csv
df.to_csv('modified.csv', index=False)
# df.to_excel('modified.xlsx', index=False)
# df.to_csv('modified.txt', index=False, sep='\t')
