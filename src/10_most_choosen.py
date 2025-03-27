import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
file_path = 'C:/Users/ebdul/OneDrive/Desktop/aaa.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Filter rows where PULocationID is 215
filtered_data_215 = data[data['PULocationID'] == 264]
#filtered_data_216 = data[data['pickup_hour'] == 18]
# Count the occurrences of each value in DOLocationID
repeated_counts = filtered_data_215['DOLocationID'].value_counts()

# Print the values sorted from most to least
print(repeated_counts)
# Calculate percentage of total trips for each destination
total_trips = repeated_counts.sum()
percentages = (repeated_counts / total_trips * 100).round(2)

# Create a DataFrame with counts and percentages
destination_stats = pd.DataFrame({
    'Count': repeated_counts,
    'Percentage': percentages
})

print("\nDestination Statistics:")
print(destination_stats)

# Plot top 10 most common destinations
plt.figure(figsize=(12, 6))
destination_stats['Count'][:10].plot(kind='bar')
plt.title('Top 10 Most Common Destinations from Location 265')
plt.xlabel('Destination Location ID')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

