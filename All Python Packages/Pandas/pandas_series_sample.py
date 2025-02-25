
#
import pandas as pd 
import numpy as np 
#create a series
s = pd.Series(np.random.randn(5))
#create a dataframe column
df = pd.DataFrame(s, columns=['column_name'])
df 

#sorting 
df.sort_values(by='column_name')

#boolean indexing
#It returns all rows in column_name,
#that are less than 10
df[df['column_name'] <= 10]

#Sample data loading
import matplotlib.pyplot as plt 
%matplotlib inline
plt.style.use('ggplot')
#read data
df = pd.read_csv('data_path_name')
#show data
df.head()

#ploting 
plt.figure(figsize=(X,Y))
#scatter
plt.scatter(x=df['column_name'].index, y=['column_name'])

#convert to datetime object
times = pd.datetimeIndex(df['time_column'])
#groupe by year, month, week, day, etc.
grouped = df.groupeby([times.year]).mean()
#plot
plt.plot(grouped['column_name'])

#access null values 
df[np.isnan(df['column_name'])]
#use forward fill gap
df['column_name'] = df['column_name'].fillna(method='ffill')
