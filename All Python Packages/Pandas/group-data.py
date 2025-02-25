import pandas as pd 
#return a groupby object, grouped by values in column named 'col'
df.groupby(by="col")
#return a groupby objec, grouped by values in index level named 'ind'
df.groupby(level="ind")
# use ? for more information