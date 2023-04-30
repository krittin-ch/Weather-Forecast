from tensorflow import keras
import joblib
import tensorflow as tf
import numpy as np

# Input data
X_data = np.array([30, 50, 100000])

name = "MLP"
model_name = "NN_" + name + "_model.h5"
scaler_name = "NN_" + name + "_scaler.pkl"

# Load model
model = keras.models.load_model(model_name)

# Load Standard Scaler
scaler = joblib.load(scaler_name)

if name == "s" :
    X_data = scaler.transform(np.concatenate(X_data.reshape(-1,3),X_data.reshape(-1,3)))
    y_data = model.predict(tf.constant(X_data))
else : 
    X_data = scaler.transform(X_data.reshape(-1,3))
    y_data = model.predict(tf.constant(X_data))

print("Prediction Value: ", int(y_data[0][0]))