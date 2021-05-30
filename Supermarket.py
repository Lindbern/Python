from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
from sklearn.datasets._samples_generator import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt

#generate blobs Amout of samples and clusters/centers, clusterd_std = amout of noice

#X,y= make_blobs(n_samples=300,centers=6,cluster_std=0.3,random_state=0)

X,y = make_moons(n_samples=300,noise=0.01)
#X,y = make_circles(n_samples=300, noise=0.01)

#KMEANS
#cluster = KMeans(n_clusters=4)
#labels = cluster.fit_predict(X) # use them as colours

"""
err=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    labels = kmeans.fit(X)
    err.append(kmeans.inertia_) # find error (Distance between of points and center, the lower the error the better the number of clusters)

plt.plot(range(1,11),err)
plt.show()

"""
"""#MEANSHIFT
bandwidth = estimate_bandwidth(X)

cluster = MeanShift(bandwidth) #keyproblem: define a proper bandwidth -> sensitve to bandwith. In sklearn= it estimates proper bandwith
labels = cluster.fit_predict(X)"""



"""#DBSCAN
clustering = DBSCAN(eps=0.115) #maximum distance
labels = cluster.fit_predict(X)"""

#HC
clustering = AgglomerativeClustering(2, linkage="single",affinity='euclidean')
labels = clustering.fit_predict(X)

#this is just to plot
plt.scatter(X[:,0], X[:,1],c=labels,cmap='plasma')
plt.show()
