import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os

# Set environment variable to avoid memory leak warning on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '1'

# Suppress warning about the change in default value of `n_init`
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster')

# Read the data from CSV into a pandas DataFrame
file_path = 'PickedEvents13Jul2023.csv'
df = pd.read_csv(file_path)

# Extract the numerical columns for clustering
numerical_columns = df[['x', 'y', 'z']]  # Add more columns as needed

# Impute missing values with the mean of the respective columns
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(numerical_columns)

# Normalize the data
normalized_data = (imputed_data - imputed_data.mean()) / imputed_data.std()

# K-means Clustering
num_clusters_kmeans = 3  # Set the number of clusters you want to create
kmeans_model = KMeans(n_clusters=num_clusters_kmeans, n_init=10)  # Explicitly set n_init
kmeans_labels = kmeans_model.fit_predict(normalized_data)

# Adding cluster labels back to the DataFrame
df['KMeans_Cluster'] = kmeans_labels

# Save the DataFrame with cluster labels to a new CSV file
output_file_path = 'PickedEvents13Jul2023_Clustered.csv'
df.to_csv(output_file_path, index=False)

# 3D Visualization of Clustering Results (assuming 'x', 'y', and 'z' are the columns)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for K-means Clustering
sc1 = ax.scatter(df['x'], df['y'], df['z'], c=kmeans_labels, cmap='rainbow', label='K-means', marker='o')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('K-means Clustering Results', fontsize=16)
ax.legend(loc='upper left', fontsize=10)

# Add a colorbar to the plot for better understanding of cluster labels
cbar = fig.colorbar(sc1, ax=ax)
cbar.ax.set_ylabel('Cluster Labels', fontsize=12)

plt.show()
