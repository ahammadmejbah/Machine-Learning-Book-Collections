import numpy as np 
#one dimensional array
np.array([1,2,3])
#two dimensional array
np.array([(1,2,3),(4,5,6)])
#1D array of length 3 all values 0
np.zeros(3)
#3x4 array with all values 1
np.ones((3,4))
#5x5 array of 0 with 1 on diagonal
np.eye(5)
#array of 6 evenly divided values from 0 to 100
np.linspace(0,100,6)
#array of values from 0 to less than 10 with step 3
np.arange(0,10,3)
#2x3 array with all values 8
np.full((2,3), 8)
#4x5 array of random floats between 0-1
np.random.rand(4,5)
#6x7 array of random floats between 0-100
np.random.rand(6,7)*100
#2x3 array with random ints between 0-4
np.random.randint(5,size=(2,3))
