# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Caminho relativo à pasta do script atual
current_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(current_dir, 'dataset')

# Dataset for CPU and GPU
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir, # root directory of the images
    labels ='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256),
    class_names=['cpu', 'gpu'],
)

# Basic CNN
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    Conv2D(32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax'), # 2 classes (CPU and GPU)
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_ds, epochs = 10, batch_size = 32)

# Visualize accuracy and loss during training
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()