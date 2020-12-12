# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:07:49 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("Iris.csv",index_col=0)
df.head(10)

df.tail(10)
df.shape

df.describe()

print(df['Species'].unique())

#performing Multivariate Analysis On Iris Dataset.
sns.pairplot(df, hue='Species', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)
plt.show()


x = df.iloc[:, [0, 1, 2, 3]].values
Sum_of_squared_distances=[]
K=range(1,15)
for k in K:
    km=KMeans(n_clusters=k)
    km=km.fit(x)
    Sum_of_squared_distances.append(km.inertia_)
    
print(Sum_of_squared_distances)
#Plotting the elbow curve by k number of clusters
    

plt.plot(K,Sum_of_squared_distances, 'bx-')   
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal K')
plt.show()
    
#In the plot, the elbow is at k=3 indicating that the optimal k for the dataset is 3
#Clustering the dataset withk=3
km1=KMeans(n_clusters=3)
km1=km1.fit(x)
print(km1.labels_)
len(km1.labels_)

#How many observations are there in knn
print(km1.labels_)
result=km1.labels_
result=pd.DataFrame(result, columns=['cluster'])    
result.groupby('cluster').size()


#The centroid of cluster
centroids=km1.cluster_centers_   
centroids=pd.DataFrame(centroids,
       columns=['Centroid_Sepellengthcm','Centroid_Sepelwidthcm','Centroid_petallenth','centroid_petalwedth'])
centroids  
    

#Predict cluster for a specie
prediction=km1.predict([[3.5,1.5,3.8,2.2]])
print(prediction)
print(km1.labels_)
#Draw the centroids ofthecluster
# Visualising the clusters - On the first two columns
plt.scatter(x[km1.labels_ == 0, 0], x[km1.labels_ == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[km1.labels_ == 1, 0], x[km1.labels_ == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[km1.labels_ == 2, 0], x[km1.labels_ == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(km1.cluster_centers_[:, 0], km1.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.legend()

    
    
    
    
    
    
    
    