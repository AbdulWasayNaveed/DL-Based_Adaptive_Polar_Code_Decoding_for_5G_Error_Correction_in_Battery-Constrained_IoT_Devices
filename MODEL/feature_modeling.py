import pandas as pd
from sklearn.utils import shuffle

# Load your dataset
filename = 'nrf24l01_balanced_data.csv'
data = pd.read_csv(filename)

# Shuffle the dataset
shuffled_data = shuffle(data, random_state=42)

# Save the shuffled dataset to a new file
shuffled_filename = 'nrf24l01_shuffled_data.csv'
shuffled_data.to_csv(shuffled_filename, index=False)

print(f"Shuffled dataset saved as {shuffled_filename}")
