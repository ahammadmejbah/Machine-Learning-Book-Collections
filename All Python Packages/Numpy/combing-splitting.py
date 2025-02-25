import numpy as np 
#adds arr2 as rows to the end of arr1
np.concatenate((arr1, arr2), axis=0)
#adds arr2 as columns to end of arr1
np.concatenate((arr1, arr2), axis=1)
#splits arr into 3 sub-arrays 
np.split(arr, 3)
#splits arr horizontally on the 5th index
np.hsplit(arr, 5)

