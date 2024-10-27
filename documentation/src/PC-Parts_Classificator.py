# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

# Path to the current script folder
current_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(current_dir, 'dataset')

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224),  # changing image size to 224x224
    validation_split=0.2,  # 20% for validation
    subset='training',
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Cache, shuffle, and prefetch data to optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Apply data augmentation only to training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Functional API for the model
input_tensor = Input(shape=(224, 224, 3))  # Define input shape
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
base_model.trainable = False  # Freeze base layers

# Define the top layers for classification
x = base_model.output
x = Flatten()(x)  # Flatten the output from VGG16
x = Dense(128, activation='relu')(x)  # Dense layer
x = Dense(64, activation='relu')(x)
output_tensor = Dense(10, activation='softmax')(x)  # Output layer for 10 classes

# Combine base model and top layers into a new model
model = Model(inputs=base_model.input, outputs=output_tensor)

# Unfreezing a few of the last layers of the VGG16
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the model
model.save('VGG16_model.keras')

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(val_ds)
print(f'Validation Accuracy: {val_acc:.2f}')

# Visualize accuracy and loss during training
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()

# Assuming that 'val_ds' has the same labels as 'train_ds'
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_ds.class_names,
            yticklabels=val_ds.class_names)
plt.xlabel("Rótulos Preditos")
plt.ylabel("Rótulos Verdadeiros")
plt.show()