# DL Reference — Keras/TensorFlow Training Loop Template
# CPU-friendly. Swap in your own model/data.
# ~80 MB RAM, <5s on CPU (tiny demo data)

import numpy as np
import matplotlib.pyplot as plt

# Suppress TF warnings
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow {tf.__version__}")
np.random.seed(42); tf.random.set_seed(42)

# ─── 1. DATA ─────────────────────────────────────────────────────────────────
# Replace with your own:
# (X_train, y_train), (X_val, y_val) = load_your_data()

N_TRAIN, N_VAL = 800, 200
X = np.random.randn(N_TRAIN + N_VAL, 20).astype(np.float32)
y = (X[:, :5].sum(1) > 0).astype(np.int32)
X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
X_val,   y_val   = X[N_TRAIN:], y[N_TRAIN:]

# ─── 2. MODEL (Sequential API) ────────────────────────────────────────────────
def build_model(in_dim, n_classes):
    return keras.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])

model = build_model(20, 2)
model.summary()

# ─── 3. COMPILE ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ─── 4. CALLBACKS ─────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "best_keras_model.keras", save_best_only=True, monitor="val_loss", verbose=0
    ),
]

# ─── 5. TRAINING ──────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

# ─── 6. EVALUATE ──────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nBest val_acc = {val_acc:.4f}  val_loss = {val_loss:.4f}")

# ─── 7. PLOT HISTORY ─────────────────────────────────────────────────────────
def plot_history(hist, save_path="keras_training.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist.history["loss"],     label="Train loss")
    axes[0].plot(hist.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(hist.history["accuracy"],     label="Train acc")
    axes[1].plot(hist.history["val_accuracy"], label="Val acc")
    axes[1].set_title("Accuracy"); axes[1].set_ylim(0,1)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=80)
    print(f"Saved {save_path}")

plot_history(history)

# ─── 8. SAVE & RELOAD ────────────────────────────────────────────────────────
model.save("final_keras_model.keras")
reloaded = keras.models.load_model("final_keras_model.keras")
_, acc2 = reloaded.evaluate(X_val, y_val, verbose=0)
print(f"Reloaded model val_acc = {acc2:.4f}")

# ─── Functional API example (reference) ─────────────────────────────────────
print("\n--- Functional API ---")
inputs  = keras.Input(shape=(20,))
x       = layers.Dense(64, activation="relu")(inputs)
x       = layers.Dropout(0.2)(x)
x       = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(2, activation="softmax")(x)
func_model = keras.Model(inputs, outputs, name="functional_mlp")
func_model.summary()
