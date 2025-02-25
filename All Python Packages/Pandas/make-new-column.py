import pandas as pd 
#compute and append one or more new columns
df.assign(Area=lambda df: df.Length*df.Height)
#add single column
df['Volume'] = df.Length*df.Height*df.Depth
#bin column into a buckets
pd.qcut(df.col, n, labels=False)
# use ? for more information