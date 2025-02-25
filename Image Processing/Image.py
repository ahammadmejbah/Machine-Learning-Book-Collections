def mean_squred_error(image01, image02):
    
    """ 
 error = np.sum((image01.astype("float") - image02.astype("float"))**2)
    error = error/float(image01.shape[0] * image02.shape[1])
    return error

"""
    
    error = np.sum((image01.astype("float") - image02.astype("float"))**2)
    error = error/float(image01.shape[0] * image02.shape[1])
    return error

def image_comparision(image01, image02):
    m = mean_squred_error(image01, image02)
    s = ssim(image01, image02)
    print("Mean Squared Error is {}\nStructural Similarity Index Measure is: {}".format(m, s))