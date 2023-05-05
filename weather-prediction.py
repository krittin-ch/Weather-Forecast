import serial
import joblib
import tensorflow as tf
from tensorflow import keras
import numpy as np

name = "s"
model_name = "NN_" + name + "_model.h5"
scaler_name = "NN_" + name + "_scaler.pkl"

# Load model
model = keras.models.load_model(model_name)

# Load Standard Scaler
scaler = joblib.load(scaler_name)

# Open serial port
ser = serial.Serial('COM8', 9600) # Replace 'COM8' with the name of your Mega 2560's serial port

while True:
    # Read data from serial port
    data = ser.readline().strip().decode()

    # Parse data
    try:
        # Split data into 3 floating-point numbers
        x1,x2,x3 = map(float, data.split(','))
        if name == 'MLP' or name == 'CNN1D':
            X_data = [[x1,x2,x3]] 
        elif name == 's':
            X_data = [[x1,x2,x3],[x1,x2,x3]] 
        # Scale input data using scaler
        X_data = scaler.transform(X_data)

        # Make prediction using model
        y_pred = model.predict(tf.constant(X_data))

        # Print prediction
        print("Prediction Value:", y_pred[0][0])

    except ValueError:
        print("Received data is not valid:", data)
