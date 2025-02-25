import pandas as pd 
#most pandas methods return a DataFrame so that
#another pandas method can be applied to the result.
#this improves readability of code
df = (pd.melt(df)
	  .rename(columns={
	  				'variable':'var',
	  				'value':'val'})
	  .query('val >= 200')
	  )
pd.melt?