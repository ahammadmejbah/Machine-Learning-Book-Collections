import pandas as pd 
#join matching rows from bdf to adf
pd.merge(adf, bdf,
		how='left', on='x1')
#join matching rows from adf to bdf
pd.merge(adf, bdf,
		how='right', on='x1')
#join data. retain only rows in both sets
pd.merge(adf, bdf,
		how='inner', on='x1')
#join data. retain all values, all rows
pd.merge(adf, bdf,
		how='outer', on='x1')
#all rows in adf that have a match  in bdf
adf[adf.x1.isin(bdf.x1)]
#all ros in adf that do not have a match in bdf
adf[~adf.x1.isin(bdf.x1)]
#rows tha appear in both ydf and xdf (intersection)
pd.merge(ydf,zdf)
#rows that appear in either or both ydf and zdf (union)
pd.merge(ydf,zdf, how='outer')
#rows tha appear in ydf but not xdf (setdiff)
pd.merge(ydf,zdf, how='outer',
		indicator=True)
.query('_merge == "left_only"')
.drop(['_merge'], axis=1)
# use ? for more information