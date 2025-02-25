import numpy as np 
#copies arr to new memory
np.copy(arr)
#creates view of arr elements with type dtype
arr.view(dtype)
#sorts arr
arr.sort()
#sorts specific axis of arr
arr.sort(axis=0)
#flattens 2D array two_d_array to 1D
two_d_arr.flatten()
#transposes arr (rows become columns and vice versa)
arr.T 
#reshapes arr to 3 rows, 4 columns without changing data
arr.reshape(3,4)
#changes arr shape to 5x6 all fills new views with 0
arr.resize((5,6))
