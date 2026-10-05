"""One-off: convert models/lstm_s*.keras to models/lstm_s*.npz (needs TensorFlow). train_pipeline.py now does this automatically."""
import json
import numpy as np
import tensorflow as tf
for s in json.load(open("models/meta.json"))["seeds"]:
    m = tf.keras.models.load_model(f"models/lstm_s{s}.keras")
    k, r, b = m.get_layer("emb").get_weights()
    np.savez(f"models/lstm_s{s}.npz", kernel=k, recurrent=r, bias=b)
    print("exported seed", s)
