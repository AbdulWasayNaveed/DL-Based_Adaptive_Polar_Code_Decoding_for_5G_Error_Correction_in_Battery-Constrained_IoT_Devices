import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import sys
import io

# Force stdout encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model
model = load_model('nrf24l01_fnn_model.h5')

# Load the dataset
data = pd.read_csv('nrf24l01_shuffled_data.csv', encoding='utf-8')

# Select the unseen samples (8001 to 10000)
X_unseen = data.iloc[8000:10000, :-1].values  # Rows 8001 to 10000 for features
y_unseen = data.iloc[8000:10000, -1].values   # Rows 8001 to 10000 for target

# Encode categorical labels (SC, SCL, LV) to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_unseen)

# Convert labels to one-hot encoding (softmax output)
y_one_hot = to_categorical(y_encoded, num_classes=3)

# Feature scaling (apply the same scaler used in training)
scaler = StandardScaler()
X_unseen = scaler.fit_transform(X_unseen)

# Evaluate the model on the unseen data
test_loss, test_accuracy = model.evaluate(X_unseen, y_one_hot)

# Print the result
print(f'Test accuracy on unseen samples (8001 to 10000) from model evaluation: {test_accuracy*100:.2f}%')

# If you want to get predictions for the unseen data, use model.predict()
predictions = model.predict(X_unseen)

# Convert predictions from one-hot encoded format to label
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Print the predictions for the unseen samples
print('Predicted labels for unseen samples (8001 to 10000):')
print(predicted_labels)

# Manually calculate accuracy by comparing predicted labels with actual labels
correct_predictions = np.sum(predicted_labels == y_unseen)
accuracy_manual = correct_predictions / len(y_unseen) * 100

# Print the manually calculated accuracy
print(f'Manually calculated accuracy on unseen samples (8001 to 10000): {accuracy_manual:.2f}%')

# Calculate the confusion matrix
cm = confusion_matrix(y_unseen, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Get the classification report (precision, recall, f1-score)
report = classification_report(y_unseen, predicted_labels, target_names=label_encoder.classes_)
print("Classification Report:")
print(report)
