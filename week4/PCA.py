from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_linnerud
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy.linalg as LA
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

linnerud_data = load_linnerud()

D = linnerud_data['data']
print(D.shape)
print(np.cov(D.T, ddof=1))

var_thresh = 100
sel = VarianceThreshold(threshold=var_thresh)
sel.fit_transform(D)

normalized_D = MinMaxScaler().fit_transform(D)
print(normalized_D)
print("Convariance: ", np.cov(normalized_D.T, ddof=1))

var_thresh = 0.06
feature_selection = VarianceThreshold(threshold=var_thresh)
feature_selection.fit_transform(normalized_D)

D_iris = load_iris()['data']
print(load_iris().feature_names)
print(D_iris.shape)
print(np.cov(D_iris.T))
print(np.cov(MinMaxScaler().fit_transform(D_iris).T))

D = D_iris[:, 2:4]
fit = plt.figure(figsize=(15,10))
plt.scatter(D[:,0], D[:,1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of data matrix D')
#plt.show()

print(D.shape)
multivariate_mean = np.mean(D, axis=0)
print(multivariate_mean)
centered_data = D - multivariate_mean
print(centered_data)
print(np.mean(centered_data, axis=0))

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.scatter(D[:,0], D[:,1], s=10, c='b', marker='s', label='original data')
ax.scatter(centered_data[:,0], centered_data[:,1], s=10, c='r', marker='x',label='centered data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper left')
plt.title("Scatter plot of centered and original data")
#plt.show()

# basis
A = np.array([[2,1],[1,-1]])
print(A)
print(D[0,:])
print(A.dot(D[0,:]))
print(A.dot(D[1,:]))
print(D.T.shape)

linearly_transformed_data = A.dot(D.T)
print(linearly_transformed_data)
print(linearly_transformed_data.shape)

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.scatter(linearly_transformed_data[0,:], linearly_transformed_data[1,:], s=10,c='r',marker='x',label='linearly transformed data')
ax.scatter(D[:,0], D[:,1], s=10, c='b', marker='s',label='original data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='lower right')
plt.title("Scatter plot of original data and linearly transformed data")
#plt.show()

Sigma = np.cov(D.T, ddof=1)
print(Sigma)
n = D.shape[0]
print(1/(n-1)*np.dot(centered_data.T, centered_data))
evalues, evectors = LA.eig(Sigma)
print("evalues: ", evalues)
print(evectors)
print(np.diag(Sigma))
total_variance = sum(np.diag(Sigma))
print("Total variance: ", total_variance)
print("Total of evalues: ", sum(evalues))

print("fraction of total variance", evalues[0]/(evalues[0]+evalues[1]))

coordinates_of_data_along_evector1 = evectors[:,0].T.dot(centered_data.T)
print(coordinates_of_data_along_evector1.shape)
print(coordinates_of_data_along_evector1)



pca = PCA(n_components=2)
pca_transformed_D = pca.fit_transform(D)
print(pca.components_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

D_iris = load_iris()['data']
pca = PCA(n_components=3)
pca_transformed_D_iris = pca.fit_transform(D_iris)
print(pca.explained_variance_ratio_)

fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(pca_transformed_D_iris[:,0], pca_transformed_D_iris[:,1], pca_transformed_D_iris[:,2], c='r')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('PCA-tranformed iris data using 3 principla components')
plt.savefig('pca-3d-tranformed.png')
plt.show()

