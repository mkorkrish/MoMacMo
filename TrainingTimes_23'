import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

# Read the CSV file into a DataFrame
df = pd.read_csv('TrainingTimes.csv')

######################
# 1. Descriptive Statistics
######################

# Select channels from Chan_11 to Chan_20
channels = df.loc[:, 'Chan_11':'Chan_20']

# Calculate descriptive statistics for the selected channels
descriptive_stats = channels.describe()

# Display the descriptive statistics
print("Descriptive Statistics:")
print(descriptive_stats)
print()


######################
# 2. Time Series Analysis
######################

# Create subplots for time series analysis
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 4), sharex=True)

# Plot time series data for each channel
for i, ax in enumerate(axes.flat):
    channel = channels.iloc[:, i]
    ax.plot(channel, color='b', linewidth=0.8)
    ax.set_title(f'Channel {i+11}', fontsize=10)
    ax.tick_params(axis='both', which='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Value', fontsize=8)

# Adjust layout and display the figure
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.suptitle('Time Series Analysis', fontsize=12)
plt.show()


######################
# 3. Correlation Analysis
######################

# Calculate the correlation matrix for the selected channels
correlation_matrix = channels.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Correlation')
plt.title('Correlation Matrix')
plt.xticks(ticks=range(channels.shape[1]), labels=channels.columns, fontsize=8, rotation=90)
plt.yticks(ticks=range(channels.shape[1]), labels=channels.columns, fontsize=8)
plt.xlabel('Channels', fontsize=10)
plt.ylabel('Channels', fontsize=10)
plt.show()


######################
# 4. Hypothesis Testing
######################

# Example: Perform t-tests to compare means of two groups (time points)
time_point_0 = df['Chan_11']
time_point_1 = df['Chan_12']
t_statistic, p_value = stats.ttest_ind(time_point_0, time_point_1)

# Display the t-statistic and p-value
print("Hypothesis Testing:")
print(f"Comparison between Channel 11 and Channel 12:")
print(f"T-statistic: {t_statistic}\nP-value: {p_value}")
print()


######################
# 5. Data Visualization
######################

# Example: Create a scatter plot for two channels
channel_1 = df['Chan_11']
channel_2 = df['Chan_12']
plt.scatter(channel_1, channel_2, color='b', alpha=0.5)
plt.xlabel('Channel 11', fontsize=10)
plt.ylabel('Channel 12', fontsize=10)
plt.title('Scatter Plot: Channel 11 vs Channel 12', fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


######################
# 6. Outlier Detection
######################

# Detect outliers in the selected channels
outliers = channels[channels.apply(lambda x: (x - x.mean()).abs() > 3 * x.std())]

# Display the outliers
print("Outlier Detection:")
if outliers.empty:
    print("No outliers found.")
else:
    print(outliers)
print()


######################
# 7. Principal Component Analysis (PCA)
######################

# Perform PCA on the selected channels
pca = PCA(n_components=2)
pca_result = pca.fit_transform(channels)

# Create a scatter plot of the PCA results
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='b', alpha=0.5)
plt.xlabel('Principal Component 1', fontsize=10)
plt.ylabel('Principal Component 2', fontsize=10)
plt.title('PCA Scatter Plot', fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
