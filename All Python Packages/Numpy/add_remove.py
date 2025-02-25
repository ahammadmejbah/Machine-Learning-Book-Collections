import numpy as np 
#appends values to end of arr
np.append(arr, values)
#inserts values into arr before index 2
np.insert(arr, 2, values)
#deletes row on index 3 of arr
np.delete(arr, 3, axis=0)
#deletes column on index 4 of arr
np.delete(arr, 4, axis=1)