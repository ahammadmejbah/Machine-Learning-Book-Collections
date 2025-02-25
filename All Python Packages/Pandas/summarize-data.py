import pandas as pd 
#count number of rows with each unique value of variable
df['w'].value_counts()
#number of rows in dataframe
len(df)
#number of distinct values in a column
df['w'].nunique()
#descriptive statistics
df.describe()
# use ? for more information