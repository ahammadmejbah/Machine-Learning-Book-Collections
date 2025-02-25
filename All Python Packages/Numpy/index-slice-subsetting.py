import numpy as np 
#return the element at index 5
arr[5]
#returns the 2D array element on index 
arr[2,5]
#assign array element on index 1 the value 4
arr[1] = 4
#assign array element on index [1][3] the value 10
arr[1,3] = 10
#return the elements at indices 0,1,2
# on a 2D array: returns rows 0,1,2
arr[0:3]
#returns the elements on rows 0,1,2, at column 4
arr[0:3, 4]
#returns the elements at indices 0,1
arr[:2]
#returns the elements at index 1 on all rows
arr[:, 1]
#returns an array with boolean values
arr < 5
#inverts a boolearn array, if its positive arr - convert to negative, vice versa
~arr
#returns array elements smaller than 5
arr[arr < 5]
