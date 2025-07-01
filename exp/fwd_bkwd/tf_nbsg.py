import tensorflow as tf
import numpy as np

# 모델 로딩
model = tf.keras.models.load_model("mobilenet_tf_savedmodel")
model.trainable = True

# 데이터 로딩
data = np.load("forward_data.npz")
x = tf.convert_to_tensor(data["x"])
y = tf.convert_to_tensor(data["y"])

# Forward 재수행 및 backward
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

with tf.GradientTape() as tape:
    logits = model(x)
    loss = loss_fn(y, logits)

grads = tape.gradient(loss, model.trainable_variables)

# gradient 저장 또는 전송
for i, g in enumerate(grads):
    print(f"Layer {i}: grad shape = {g.shape}, mean = {tf.reduce_mean(g):.4f}")
