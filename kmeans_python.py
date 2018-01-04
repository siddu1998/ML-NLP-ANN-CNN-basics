import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values #anual income and spending score

#now to find the number of clusters-->using elbow method
from sklearn.cluster import KMeans
wcss=[]
#let us take for 10 cluster
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    #intertia calculates the wcss since wcss is also called inertia
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('THe elbow method')
plt.xlabel('NUmber of clustres')
plt.ylabel('WCSS')
plt.show()

#from the graph get the elbow i.e number of clusters
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)
print(*y_kmeans,sep='\n') #five clusters-->0 to 4

#visulaizing the clusters
#this is only for 2-D incase you have more paramaters then u need more axis for that u cant execute below so make sure u delete the shit


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,color='red',label='Cluster1')
#we need to compare the earning rate vs spending rate
#what is y_kmeans --> the clusters the model has identified 0->4
#all the points with same category need to be plotted to tgether
# #scattering a point needs both x cordinate and y cordinate
#our y_means predicts based on spending score and amount earned
#X[y_kmeans==0,0] ---> the x-cordinate::  X(coloumn 0) and y_kmeans =0 --> spending score whose estimated category=0
#X[y_kmeans==0,1] ---> the y-cordinate::  X(coloumn 1) and y_kmeans =0 --> anual income whose estimated category=0
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,color='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,color='green',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,color='magenta',label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,color='cyan',label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()