import numpy as np

def getData(seed=42, N=100): # TODO : add an angle to ratate the XOR (for fun)
    np.random.seed(seed)
    D = 2
    X = np.random.random((N,2))*2-1
    yregress = X[:,0]*X[:,1] +np.random.normal(0,0.01, (N))
    yclassif = (yregress >0)*2-1
    return X, yregress, yclassif
# yregress, yclassif = createData()
