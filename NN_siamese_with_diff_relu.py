#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils import losses_utils
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Activate GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Import Dataset
df = pd.read_csv('DataSet.csv')
data_t2m = np.array(df['t2m'])
data_rh = np.array(df['rh'])
data_ptype = np.array(df['ptype'])

X_data = np.concatenate([data_t2m.reshape(-1, 1), data_rh.reshape(-1, 1)], axis=1)
y_data = data_ptype.reshape(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, shuffle=True)

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split train data
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.4, random_state=42, shuffle=True)

# Define lazy_relu and leay_relu
def lazy_relu(x):
    return tf.maximum(x, 0.25*x)

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)

# Create the first model
model1 = keras.models.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation='relu'),
    keras.layers.Dense(8, activation=leaky_relu)
])

# Create the second model
model2 = keras.models.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation=lazy_relu),
    keras.layers.Dense(8*2, activation=lazy_relu),
])

# Concatenate the two models
combined_model = keras.layers.Concatenate()([model1.output, model2.output])
z = keras.layers.Dense(4, activation="relu")(combined_model)

# Output layer
output = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.models.Model(inputs=[model1.input, model2.input], outputs=output)

# Compile the model with the appropriate loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='auto')
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

# Fit the model on the training data
history = model.fit([X_train, X_train], y_train, epochs=1, batch_size=32, validation_split=0.1)

# Plot the training history
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.legend()

# Evaluate the performance of the model on the testing set
loss, accuracy = model.evaluate([X_test,X_test], y_test)
print(f'Test loss: {loss:.2f}')
print(f'Test accuracy: {accuracy:.2f}')

# Make predictions on new data
y_pred = np.round(model.predict([X_test,X_test]))

# Confusion Matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# Save model image
plot_model(model, to_file='model_sdr.png', show_shapes=True)

# Save the scaler object
joblib.dump(scaler, 'scaler_sdr.pkl')

# Sa
model.save('NN_sdr_model.h5')