import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('trainingEventsDistributed.csv')

# Extract 'x' and 'y' columns
X = data[['x', 'y']]

# Create a KMeans instance with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)

# Fit the model to the data
kmeans.fit(X)

# Predict the clusters
data['cluster'] = kmeans.predict(X)

# Set up a color map with vibrant colors
colors = ['blue', 'green', 'red']

# Create a scatterplot of the 'x' and 'y' variables
# Color the points by the cluster assignments
plt.figure(figsize=(10, 8))
for cluster in data['cluster'].unique():
    plt.scatter(data.loc[data['cluster'] == cluster, 'x'], 
                data.loc[data['cluster'] == cluster, 'y'], 
                color=colors[cluster],
                label=f'Cluster {cluster}')

# Add a legend
plt.legend()

# Add labels for the axes
plt.xlabel('x')
plt.ylabel('y')

# Add a title for the plot
plt.title('K-Means Clustering with 3 Clusters')

# Show the plot
plt.show()
