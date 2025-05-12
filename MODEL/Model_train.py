import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
import io

# Force stdout encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the dataset
data = pd.read_csv('nrf24l01_shuffled_data.csv', encoding='utf-8')

# Use only the first 8000 samples
data = data.iloc[:8000]

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode target labels (SC, SCL, LV)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded, num_classes=3)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the model
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Early stopping on training loss (no val_loss available)
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_scaled, y_one_hot, 
                    epochs=25, 
                    batch_size=32, 
                    verbose=1, 
                    callbacks=[early_stopping])

# Save the trained model
model.save('nrf24l01_fnn_model.h5')

# Plot training loss and accuracy
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
