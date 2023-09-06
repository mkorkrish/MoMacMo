# Import necessary libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('trainingEventsDistributed.csv')

# Handle missing values by simple imputation (mean)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=[np.number])))
data_imputed.columns = data.select_dtypes(include=[np.number]).columns
data[data_imputed.columns] = data_imputed

# Choose columns to include in the clustering
columns_to_cluster = ['x', 'y', 'power']

# Extract the data for clustering
data_for_clustering = data[columns_to_cluster]

# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler()
data_for_clustering = scaler.fit_transform(data_for_clustering)

# Use the Elbow method to find a good number of clusters using WCSS
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)

    kmeans.fit(data_for_clustering)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Assume the optimal number of clusters from the Elbow method is 3
n_clusters = 3

# Define the k-means clustering model
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

# Fit the k-means clustering model and get the cluster labels
cluster_labels = kmeans.fit_predict(data_for_clustering)

# Add the cluster labels to the original data
data['cluster'] = cluster_labels

# Save the data with the new cluster labels
data.to_csv('trainingEventsDistributed_with_clusters.csv', index=False)

print("Clustering completed and data saved with cluster labels.")

# Create a scatter plot of x and y, color-coded by cluster
plt.figure(figsize=(10, 7))
plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
plt.title('Clusters (x vs y)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Cluster')
plt.show()

# Create a scatter plot of x and power, color-coded by cluster
plt.figure(figsize=(10, 7))
plt.scatter(data['x'], data['power'], c=data['cluster'], cmap='viridis')
plt.title('Clusters (x vs power)')
plt.xlabel('x')
plt.ylabel('power')
plt.colorbar(label='Cluster')
plt.show()

# Create a 3D scatter plot of x, y, and power, color-coded by cluster
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(data['x'], data['y'], data['power'], c=data['cluster'], cmap='viridis')
ax.set_title('3D Cluster Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('power')
plt.colorbar(sc, label='Cluster')
plt.show()

# Compute Silhouette Score
silhouette_avg = silhouette_score(data_for_clustering, cluster_labels)
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
