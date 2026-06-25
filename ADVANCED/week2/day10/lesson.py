# Advanced Day 10 — TensorFlow / Keras (CPU)
# ~400 MB RAM, ~2-3 min on CPU
# Mirrors Day 09 to let you compare PyTorch vs Keras side by side.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info messages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

np.random.seed(42); tf.random.set_seed(42)
print(f"TensorFlow version: {tf.__version__}")

# ─── 1. TENSORS IN TF ────────────────────────────────────────────────────────
print("\n=== 1. TensorFlow Tensors ===")
t = tf.constant([[1., 2.], [3., 4.]])
print(t); print(t.shape, t.dtype)
print(t + 1)                    # element-wise broadcast
print(t @ tf.transpose(t))      # matrix multiply

# Variables (mutable)
v = tf.Variable([1., 2., 3.])
v.assign(v + 10)
print(v)

# ─── 2. AUTOGRAD IN TF ───────────────────────────────────────────────────────
print("\n=== 2. Autograd (GradientTape) ===")
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1
grad = tape.gradient(y, x)
print(f"x=3: y={y.numpy():.2f}, dy/dx={grad.numpy():.2f}")  # 2*(3+1)=8

# ─── 3. BUILDING A MODEL — SEQUENTIAL API ────────────────────────────────────
print("\n=== 3. Sequential API ===")
model_seq = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64,  activation="relu"),
    layers.Dense(10,  activation="softmax"),
])
model_seq.summary()

# ─── 4. FUNCTIONAL API ───────────────────────────────────────────────────────
print("\n=== 4. Functional API ===")
inputs  = keras.Input(shape=(64,))
x = layers.Dense(128, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model_fn = keras.Model(inputs, outputs, name="digit_net")
model_fn.summary()

# ─── 5. COMPILE ──────────────────────────────────────────────────────────────
model_fn.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
)

# ─── 6. DATA ─────────────────────────────────────────────────────────────────
digits = load_digits()
sc = StandardScaler()
X = sc.fit_transform(digits.data).astype(np.float32)
y = digits.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ─── 7. CALLBACKS ────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

# ─── 8. TRAIN ────────────────────────────────────────────────────────────────
history = model_fn.fit(
    X_tr, y_tr,
    epochs          = 30,
    batch_size      = 32,
    validation_split= 0.15,
    callbacks       = callbacks,
    verbose         = 1,
)

# ─── 9. EVALUATE ─────────────────────────────────────────────────────────────
test_loss, test_acc = model_fn.evaluate(X_te, y_te, verbose=0)
print(f"\nTest loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

y_pred = model_fn.predict(X_te, verbose=0).argmax(axis=1)
print(classification_report(y_te, y_pred))

# ─── 10. SAVE & LOAD ─────────────────────────────────────────────────────────
model_fn.save("digit_keras_model.keras")
loaded = keras.models.load_model("digit_keras_model.keras")
print("Loaded model test acc:", loaded.evaluate(X_te, y_te, verbose=0)[1])
import os; os.remove("digit_keras_model.keras")

# ─── 11. VISUALISE ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train")
axes[1].plot(history.history["val_accuracy"], label="Val")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout(); plt.savefig("keras_digits.png", dpi=80)
print("Saved keras_digits.png")

# ─── PYTORCH vs KERAS — COMPARISON TABLE ─────────────────────────────────────
print("""
=== PyTorch vs Keras Quick Comparison ===
                 PyTorch                           Keras / TF
Define model     nn.Module subclass                keras.Sequential or Functional API
Forward pass     model(x)                          model(x) or model.predict()
Loss             criterion(out, y)                 compile(loss=...) or manual
Backprop         loss.backward()                   tape.gradient() or automatic
Param update     optimizer.step()                  automatic inside fit()
Training loop    Manual for/DataLoader             model.fit() with callbacks
Debugging        Easier (Python control flow)      Easier for quick experiments
Production       TorchServe / ONNX                 TFLite / TF Serving / SavedModel
""")
