# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Path to the current script folder
current_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(current_dir, 'dataset')

# Dataset for CPU and GPU
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir, # root directory of the images
    labels ='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256),
    validation_split=0.2, # 20% for validation
    subset='training', # training subset
    seed=123 # ensure that separation is consistent
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir, # root directory of the images
    labels ='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256),
    validation_split=0.2, # 20% for validation
    subset='validation', # validation subset
    seed=123 # ensure that separation is consistent
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
    Dense(14, activation='softmax'), # 2 classes (CPU and GPU)
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_ds, # training data
                    validation_data=val_ds,
                    epochs = 20,
                    verbose=1)

# Saving the model
model.save('pc_parts_classification_model.h5')

# Evaluating the model on validation data
val_loss, val_acc = model.evaluate(val_ds)
print(f'Acc in validation set: {val_acc:.2f}')

# Visualize accuracy and loss during training
#plt.plot(history.history['accuracy'], label='Accuracy')
#plt.plot(history.history['loss'], label='Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Metrics')
#plt.legend()
#plt.show()