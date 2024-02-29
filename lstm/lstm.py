import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load your CSV data
df = pd.read_csv('outputcp.csv')

# Assuming your target variable is in the first column
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values  # Exclude the first column as it's the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using Min-Max scaling
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Normalize the target using a separate scaler
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape the data for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model for binary classification
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy')
model.load_weights('lstm_weights.h5')
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss= model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
model.save_weights('lstm_weights.h5')
# Number of sequences to generate
num_sequences = 50

# Initialize an array to store the generated sequences
generated_sequences = []

# Use the last sequence from the test set as the initial input for generation
# Reshape the current sequence to match the shape that the model expects
current_sequence = X_test[-1].reshape(1, 1, -1)

# Generate new sequences
for _ in range(num_sequences):
    # Predict the next value in the sequence
    predicted_value = model.predict(current_sequence)
    
    # Inverse transform the predicted value
    predicted_value = scaler_y.inverse_transform(predicted_value)
    
    # Append the inverse transformed predicted value to the generated sequences
    generated_sequences.append(predicted_value[0])
    
    # Update the current sequence with the new predicted value
    # Reshape the predicted value to match the shape that the model expects
    current_sequence = np.concatenate([current_sequence[:, :, 1:], predicted_value.reshape(1, 1, -1)], axis=2)

# Convert the generated sequences to a NumPy array
generated_sequences = np.array(generated_sequences)

# Plot the generated sequences
plt.plot(generated_sequences, label='Generated Data')
plt.plot(scaler_y.inverse_transform(y_test[:50]), label='Original Data')
plt.legend()
plt.show()