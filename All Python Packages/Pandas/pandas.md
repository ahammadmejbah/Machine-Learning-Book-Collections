# Pandas Cheatsheet
1. [Reading/writing data](#readingwriting-data)
2. [Exploratory data analysis](#exploratory-data-analysis)
3. [Dataframe methods](#dataframe-methods)
4. [Data types](#data-types)

## Reading/writing data
| Methods | Description |
| --- | --- |
| `pd.read_csv(csv, index_col, header, names, na_values, parse_dates)` | reading csv file |
| `pd.DataFrame({'key':[...], 'key2':[...]})` | reading from dictionary |
| `df.to_csv(csv, sep)` | writing to csv file |
| `df.to_excel(csv)` | writing to excel file |

## Exploratory data analysis
### EDA
| Methods | Description |
| --- | --- |
| `df.head() / df.tail()` | return first/last 5 rows |
| `df.first() / df.last()` | return first/last rows |
| `df.sample(n, frac)` | randomly select n/fraction of rows |
| `df.idxmax() / df.idxmin()` | return maximum / minimum index value |
| `df.columns` | return the columns |
| `df.info()` | return the data types |
| `df.shape` | return the dimension |
| `df(.column).describe()` | return the summary statistics |
| `df.value_counts(dropna=BOOLEAN)` | return frequency count of a column |
| `df['co'l].unique()` | return values of column |
| `df['col'].count()` | return count |
| `df['col'].mean()` | return mean |
| `df.sum()` | sum of values |
| `df.cumsum()` | cumulative sum of values |
| `df.std()` | return standard deviations |
| `df.median()` | return median |
| `df.quantile()` | return quantiles |
| `df.max() / df.min()` | return maximum / minimum |

### vEDA
| Methods | Description |
| --- | --- |
| `df.plot(x, y)` | line plot |
| `df.plot(x, y, kind='scatter'\|'box')` | scatterplot/boxplot |
| `df.plot(y, kind='hist', bins, range, alpha, cumulative, normed)` | histogram customization |
| `df['col'].plot('hist')` | histogram plot for the column of the dataframe |
| `df.boxplot('col', by)` | boxplot |
| `df.loc[start:end, 'col'].plot()` | time series plot |
| `plt.plot(numpy.ndarry\|pandas.core.series.Series\|df)` | plotting arrays/series/df |
| `plt.savefig(image)` | saving plot |

## Dataframe methods
| Methods | Description |
| --- | --- |
| `df.index` | return index |
| `df.set_index(['col1', 'col2'])` | col1 is the 1st index and col2 is the 2nd index |
| `df.unstack(level='col2')` | unstack, col2 not an index anymore |
| `df.stack(level='col2')` | stack, col2 is the 2nd index again |
| `df.swaplevel(0,1)` | swap col1 and col2, col2 is the 1st index and col1 is the 2nd index |
| `df.reindex('col', method='ffill'\|'bfill')` | reindexing dataframe and fill missing values |
| `df.sort_index()` | sort by labels along an axis |
| `df.sort_values(by='Country', ascending)` | sort the values by country column |
| `df.loc[:, 'col1':'col3']` | return all rows from col1 to col3 |
| `df.loc['row1':'row3', :]` | return all cols from row1 to row3 |
| `df.loc['row1':'row3', ['col1', 'col5']]` | return row1 to row3 for col1 & col5 |
| `df.loc[start:end]` | return data based on start & end date |
| `df.loc[:, df.all()]` | select columns with all nonzeros |
| `df.loc[:, df.any()]` | select columns with any nonzeros |
| `df.loc[:, df.isnull().any()]` | select columns with any NaNs |
| `df.loc[:, df.notnull().all()]` | select columns without NaNs |
| `df.iloc[2:5, 1:]` | return row 2 to 5 from column 1 onwards |
| `df['col'].str.contains(text).sum()` | search column for number of rows containing text |
| `df['col1'] / df[['col1']]` | return series vs dataframe |
| `df[['col1', 'col2']]` | selecting only some columns |
| `df.copy()` | make a copy of dataframe |
| `pd.concat([df1, df2], ignore_index)` | combine data |
| `pd.merge(left, right, on, left_on, right_on)` | joining 2 dataframes based on column |
| `pd.melt(frame, id_vars, value_vars, var_name, value_name)` | convert columns to rows |
| `df.pivot(index, columns, values, aggfunc)` | turn unique values into columns, aggfunc will handle duplicate values |
| `df.pivot_table(index, columns, values, aggfunc)` | pivot table can contain duplicates |
| `df.groupby('col').count()` | return group count |
| `df.groupby('col')[['col1','col2']].agg(['count', 'max', 'sum'])` | return group count & max & sum |
| `df.groupby('col')[['col1','col2']].agg({'col1':'count', 'col2':'sum'})` | return group count for col1 and group sum for col2 |
| `df.groupby('col')[['col1','col2']].agg(func)` | return group values based on customized aggregation |
| `dfGroupBy.groups.keys()` | return the keys of DataFrameGroupBy |
| `df.drop(['col1', 'col2'], axis=1)` | drop col1 & col2 |
| `df.drop_duplicates()` | drop duplicates |
| `df.dropna(how='any')` | drop rows with NaNs |
| `df.fillna(method)` | fill missing values |
| `df[newColumn] = np.nan` | broadcasting NaN value to new column |
| `df.resample('D')(.mean()\|.sum()\|.max()\|.count())` | aggregating daily mean |
| `df.resample('A').interpolate('linear')` | interpolate missing data |
| `df.apply(func, axis=0\|1, pattern)` | apply function on row (0) or column (1), pattern for regex expression |
| `df.applymap(func)` | apply function element-wise |
| `df[df.col1 > 1]` | filter rows with col1 values > 1 |
| `df[(df.col1 > 1) & (df.col2 < 1)]` | filter rows where col1 > 1 and col2 < 1 |
| `df[(df.col1 > 1) \| (df.col2 < 1)]` | filter rows where col1 > 1 or col2 < 1 |

## Data types
| Methods | Description |
| --- | --- |
| `df.astype(str\|'category')` | convert data types |
| `pd.to_numeric(df['col'], errors)` | convert to integer and clean data |
| `pd.to_datetime()` | convert strings to datetime |
| `df[datetimeColumn].dt.time` | return the hour of datetime column |
| `df[datetimeColumn].dt.tz_localize('US/Central')` | set timezone of datetime column |
| `df[datetimeColumn].dt.tz_convert('US/Central')` | convert timezone of datetime column |
