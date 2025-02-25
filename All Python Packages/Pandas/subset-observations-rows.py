import pandas as pd 
#extract rows that meet logical criteria
df[df.Lenght > 7]
#remove duplicate rows (only considers columns)
df.drop_duplicates()
#select first n rows
df.head(n)
#select last n rows
df.tail(n)
#randomly select fraction of rows
df.sample(frac=0.5)
#randomly select n rows
df.sample(n=10)
#select rows by position
df.iloc[10:20]
#select and order top n entries
df.nlargest(n, 'value')
#select and order bottom n entries
df.nsmallest(n, 'value')
#refer to logic in python (and pandas)
# use ? for more information
