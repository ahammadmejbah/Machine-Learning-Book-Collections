import numpy as np 
#from a text file
np.loadtxt('file_name.txt')
#from a csv file
np.genfromtxt('file_name.csv', delimiter=',')
#writes to a text file
np.savetxt('file_name.txt', arr, delimiter=' ')
#writes to a csv file
np.savetxt('file_name.csv', arr, delimiter=',')