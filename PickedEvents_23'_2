import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set environment variable to avoid warning
os.environ["OMP_NUM_THREADS"] = "1"

# Load the dataset
df = pd.read_csv('PickedEvents13Jul2023.csv')

# Choose the features for clustering
features = ['x', 'y', 'volume', 'power', 'count']

# Standardize the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Determine the optimal number of clusters using the Elbow Method
range_n_clusters = list(range(2, 10))
ssd = [KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(df_scaled).inertia_ for num_clusters in range_n_clusters]

# Plot SSD vs. number of clusters
plt.figure(figsize=(8,6))
plt.plot(range_n_clusters, ssd, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Create a K-means model with the chosen number of clusters (4 in this case)
clusters = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(df_scaled)

# Add the cluster assignments to the original dataframe
df['Cluster'] = clusters

# Create box plots for each numerical feature
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=feature, data=df)
    plt.title(f'Box Plot of {feature.capitalize()} by Cluster')
    plt.grid(axis='y')
    plt.show()

# Create bar plots for each categorical feature
for col in ['label', 'mapType']:
    plt.figure(figsize=(8, 6))
    df.groupby(['Cluster', col]).size().unstack().plot(kind='bar', stacked=True, grid=True)
    plt.ylabel('Count')
    plt.title(f'Distribution of {col.capitalize()} Across Clusters')
    plt.show()
