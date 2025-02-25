import numpy as np
def gradientDescent(x, y):
    m_curr = b_curr = 0 
    iterations = 1000 
    n = len(x)
    learning_rate = 0.01
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr  # y = mx + c
        cost_function = (1/n)*sum([val**2 for val in (y - y_predicted)])   # mean squared error
        #derivative respect of intercept
        bd = (2/n)*sum(y - y_predicted)

        # derivative repect of scope
        md = (2/n)*sum(x*(y - y_predicted))

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        print(f"Iteration {i}, M : {m_curr}, B : {b_curr}, Cost : {cost_function}")
if __name__ == '__main__':
    x = np.array([1,2,3,4,5])
    y = np.array([5,10,15,20,25])

    gradientDescent(x, y)