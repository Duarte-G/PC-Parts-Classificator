# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping

# Path to the current script folder
current_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(current_dir, 'dataset')

# Data augmentation layers
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir, # root directory of the images
    labels ='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224), # changing image size to 224x224
    validation_split=0.2, # 20% for validation
    subset='training', # training subset
    seed=123 # ensure that separation is consistent
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir, # root directory of the images
    labels ='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224), # changing image size to 224x224
    validation_split=0.2, # 20% for validation
    subset='validation', # validation subset
    seed=123 # ensure that separation is consistent
)

# Cache, shuffle, and prefetch data to optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load pre-trained VGG16 without top
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# Freeze the VGG16 base layers so they can't be trained
base_model.trainable = False

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Pre-trained model
model = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    base_model,  # VGG16 without top
    GlobalAveragePooling2D(),  # Pooling global para reduzir as dimensões
    Dense(128, activation='relu'),  # Additional dense layer
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])
# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_ds, # training data
                    validation_data=val_ds,
                    epochs = 40,
                    verbose=1,
                    callbacks=[early_stopping])

# Saving the model
model.save('pc_parts_classification_model.keras')

# Evaluating the model on validation data
val_loss, val_acc = model.evaluate(val_ds)
print(f'Acc in validation set: {val_acc:.2f}')

# Visualize accuracy and loss during training
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()