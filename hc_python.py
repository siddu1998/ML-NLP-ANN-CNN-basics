import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values #anual income and spending score

#plotting dendogram
import scipy.cluster.hierarchy as sch
#ward-->minimize variance
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendagram')
plt.xlabel('Customes')
plt.ylabel('Eucledian Distances')
plt.show()
#from the graph we get 5 clusters

#fiiting hc with the data set
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)
print(y_hc)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,color='red',label='Cluster1')
#we need to compare the earning rate vs spending rate
#what is y_hc --> the clusters the model has identified 0->4
#all the points with same category need to be plotted to tgether
# #scattering a point needs both x cordinate and y cordinate
#our y_means predicts based on spending score and amount earned
#X[y_hc==0,0] ---> the x-cordinate::  X(coloumn 0) and y_hc =0 --> spending score whose estimated category=0
#X[y_hc==0,1] ---> the y-cordinate::  X(coloumn 1) and y_hc =0 --> anual income whose estimated category=0
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,color='blue',label='Cluster2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,color='green',label='Cluster3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,color='magenta',label='Cluster4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,color='cyan',label='Cluster5')
# plt.scatter(hc.cluster_centers_[:,0],hc.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()




