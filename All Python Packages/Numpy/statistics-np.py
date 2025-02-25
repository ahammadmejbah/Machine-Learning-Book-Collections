import numpy as np 
#returns mean along specific axis
np.mean(arr, axis=0)
#returns sum of arr
arr.sum()
#returns minimum value of arr
arr.min()
#returns maximum value of specific axis
arr.max(axis=0)
#returns the variance of array
np.var(arr)
#returns the standard deviation of specific axis
np.std(arr, axis=1)
#returns correlation coefficient of array
arr.corrcoef() 