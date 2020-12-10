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

df=pd.read_csv("Iris.csv")
df.head(10)

df.tail(10)

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
    
    
#Plotting the elbow curve by k number of clusters
    

plt.plot(K,Sum_of_squared_distances, 'bx-')   
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal K')
plt.show()
    
#In the plot, the elbow is at k=3 indicating that the optimal k for the dataset is 3
#Clustering the dataset withk=3
knn=KMeans(n_clusters=3)
knn=knn.fit(x)
print(knn.labels_)

#How many observations are there in knn
print(knn.labels_)
result=knn.labels_
result=pd.DataFrame(result, columns=['cluster'])    
result.groupby('cluster').size()


#The centroid of cluster
centroids=knn.cluster_centers_   
centroids=pd.DataFrame(centroids,
        columns=['Centroid_Setosa','Centroid_Verginica','Centroid_Versicolor','Centroid_Species'])
centroids  
    

#Predict cluster for a specie
prediction=knn.predict([[3.5,1.5,3.8,2.2]])
print(prediction)

#Draw the centroids ofthecluster
# Visualising the clusters - On the first two columns
plt.scatter(x[knn.labels_ == 0, 0], x[knn.labels_ == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[knn.labels_ == 1, 0], x[knn.labels_ == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[knn.labels_ == 2, 0], x[knn.labels_ == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(knn.cluster_centers_[:, 0], knn.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')


    
    
    
    
    
    
    
    