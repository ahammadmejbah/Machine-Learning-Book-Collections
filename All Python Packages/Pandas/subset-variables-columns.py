import pandas as pd 
#select multiple columns with specific names
df[['width', 'col_name2','col_name3']]
#select single column with specific name
df['width'] #or
df.width
#select columns whose names matches regular expression regex
df.filter(regex='regex')
#select all columns between x2 and x4 inclusive
df.loc[:, 'x2':'x4']
#select columns in positions 1,2 and 5.
df.iloc[:,[1,2,5]]
#select rows meeting logical condition, and only the specific columns
df.loc[df['a'] > 10, ['a','c']]
# use ? for more information