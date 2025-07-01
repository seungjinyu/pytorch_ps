import tensorflow as tf
import numpy as np
import os

# 고정 시드
tf.random.set_seed(42)

# 모델 정의
model = tf.keras.applications.MobileNetV2(input_shape=(64, 64, 3), weights=None, classes=10)
model.trainable = True

# Dummy input
x = tf.random.normal((2, 64, 64, 3))
y = tf.constant([1, 0])

# Loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Forward pass
with tf.GradientTape() as tape:
    logits = model(x)
    loss = loss_fn(y, logits)

# Save tape and x/y/logits (or send over network)
grads = tape.gradient(loss, model.trainable_variables)

# 모델 저장
model.save("mobilenet_tf_savedmodel")
np.savez("forward_data.npz", x=x.numpy(), y=y.numpy(), logits=logits.numpy())
