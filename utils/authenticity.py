import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

model = load_model("mesonet_model.h5")

def check_face_authenticity(frame):
    face = cv2.resize(frame, (256, 256))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    pred = model.predict(face, batch_size=1)[0][0]
    return "REAL" if pred > 0.5 else "FAKE"

