#from matplotlib.mlab import PCA
import matplotlib.pylab as MPL
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy
#data = array(randint(10,size=(10,3)))
#results = PCA(data)
def PCA2(data, dims_rescaled_data):
    pca = PCA(n_components=dims_rescaled_data)
    return pca.fit_transform(data)

"""
def PCA1(data, dims_rescaled_data):
    
    #returns: data transformed in 2 dims/columns + regenerated original data
    #pass in: data as 2D NumPy array
    
    import numpy as NP
    from scipy import linalg as LA
    mn = NP.mean(data, axis=0)
    sd = NP.std(data, axis=0)
    # mean center the data
    #data -= mn
    #data /= sd
    # calculate the covariance matrix
    C = NP.cov(data.T)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = LA.eig(C)
    # sorted them by eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:,:dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    data_rescaled = NP.dot(evecs.T, data.T).T
    # reconstruct original data array
    #data_original_regen = NP.dot(evecs, dim1).T + mn
    return data_rescaled


def plot_pca(data):
    clr1 =  '#2026B2'
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc = PCA2(data,2)
    ax1.plot(data_resc[:,0], data_resc[:,1], '.', mfc=clr1, mec=clr1)
    print data_resc[:,0].shape
    print data.shape
    MPL.show()

"""
#iris = datasets.load_iris()
#mydata = iris.data
#print mydata
#plot_pca(mydata)
