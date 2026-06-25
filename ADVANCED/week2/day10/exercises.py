# Advanced Day 10 — Exercises
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
np.random.seed(42); tf.random.set_seed(42)

# Exercise 1 — Regression in Keras
# from sklearn.datasets import make_regression
# Build a Keras network for regression (output layer: 1 unit, linear activation).
# Loss = MSE. Metric = MAE. Train 20 epochs. Plot train/val loss.
# TODO

# Exercise 2 — Custom training loop in TF
# Replicate the Keras fit() loop manually using tf.GradientTape:
# for each batch: tape → loss → gradients → optimizer.apply_gradients
# Train for 10 epochs on the digits data. Print loss each epoch.
# TODO

# Exercise 3 — Regularization comparison
# Build 3 identical digit networks (2 hidden layers, 128 neurons each) with:
# a) No regularization
# b) Dropout(0.5) after each hidden layer
# c) L2 regularization (kernel_regularizer=keras.regularizers.l2(0.001))
# Train each for 20 epochs. Plot val_accuracy side by side.
# TODO

# Exercise 4 — Model subclassing
# Create a Keras model by subclassing keras.Model (like nn.Module in PyTorch):
# class MyModel(keras.Model):
#   def __init__(self): build layers
#   def call(self, x): define forward pass
# Use it on the digits classification task.
# TODO

# Exercise 5 — Learning rate schedule
# Use keras.optimizers.schedules.CosineDecay to create a LR schedule.
# Start at 1e-3, decay over 500 steps. Plot the LR schedule.
# Train digit classifier with this LR and compare to constant LR.
# TODO
