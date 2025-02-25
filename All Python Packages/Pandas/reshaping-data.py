import pandas as pd 
#gather columns into rows
pd.melt(df)
#spread rows into columns
df.pivot(columns='var', values='val')
#append rows of dataframe
pd.concat([df1,df2])
#append columns of dataframe
pd.concat([df1,df2], axis=1)
#order rows by values of a column (low to high)
df.sort_values('row_name')
#order row by values of a column (high to low)
df.sort_values('row_name', ascending=False)
#return the columns of a dataframe
df.rename(columns={'y':'year'})
#sort the index of a dataframe
df.sort_index()
#reset index of dataframe to row numbers, moving index to column
df.reset_index()
#drop columns from dataframe
df.drop(['Length','Height'], axis=1)
#use ? for more info
