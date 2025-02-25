
# coding: utf-8

# ## Uber Data Analysis 
# 
# I'll be showing you how to analyze some real data, in this case we are going to do some analysis on Uber data.
# 
# I am gonna show you how to graph something like Location data, you'll be apply to produce the image below after you finished this tutorial:
# 
# <img style="float:left;" src="https://i.imgur.com/fWM9sw9.png"></img>

# In[1]:


import pandas as pd
import numpy as np


# In[30]:


data = pd.read_csv("data/uber_data.csv")


# In[31]:


data.head()


# In[32]:


data['Date/Time'] = data['Date/Time'].map(pd.to_datetime)


# In[33]:


data.tail()


# In[34]:


# let's get the days of the month 
def day_of_month(dt):
    return dt.day

data['DayOM'] = data['Date/Time'].map(day_of_month)


# In[35]:


data.head()


# In[36]:


def week_day(dt):
    return dt.weekday()

data['WeekDay'] = data['Date/Time'].map(week_day)


# In[37]:


def get_hour(dt):
    return dt.hour


# In[38]:


data['hour'] = data['Date/Time'].map(get_hour)

data.tail()


# In[41]:


#let's do some analysis
import matplotlib.pyplot as plt
plt.hist(data.DayOM, bins=30, rwidth=.8, range=(0.5, 30.5))
plt.xlabel('date of the month')
plt.ylabel('frequency')
plt.title('Frequency by D-o-M - uber - Apr 2014')
plt.show()


# In[43]:


def count_rows(rows):
    return len(rows)

by_date = data.groupby('DayOM').apply(count_rows)
by_date.head()


# In[46]:


plt.bar(range(1, 31), by_date)
plt.show()


# In[48]:


by_date_sorted = by_date.sort_values()
by_date_sorted.head()


# In[49]:


plt.bar(range(1, 31), by_date_sorted)
plt.xticks(range(1,31), by_date_sorted.index)
plt.xlabel('date of the month')
plt.ylabel('frequency')
plt.title('Frequency by DoM - uber - Apr 2014')
plt.show()
("")


# In[52]:


# analysis for week day
plt.hist(data.WeekDay, bins=7, range =(-.5,6.5), rwidth=.8, color='green', alpha=.4)
plt.xticks(range(7), 'Mon Tue Wed Thu Fri Sat Sun'.split())
plt.show()


# In[57]:


# plot location
plt.hist(data['Lon'], bins=100, range = (-74.1, -73.9), color='purple', alpha=.5, label = 'longitude')
plt.grid()
plt.legend(loc='upper left')
plt.twiny()
plt.hist(data['Lat'], bins=100, range = (40.5, 41), color='g', alpha=.5, label = 'latitude')
plt.legend(loc='best')
plt.show()
("")


# In[56]:


plt.figure(figsize=(20, 20))
plt.plot(data['Lon'], data['Lat'], '.', ms=1, alpha=.5, color='purple')
plt.xlim(-74.2, -73.7)
plt.ylim(40.7, 41)
plt.show()


# # more to come, soon!
