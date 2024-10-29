# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Path to the current script folder
current_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(current_dir, 'dataset')

# Get class names from directory
class_names = sorted(os.listdir(dataset_dir))

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# Preprocess
def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image, label

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size= 32,
    image_size=(224, 224), # changing image size to 224x224
    validation_split=0.2, # 20% for validation
    subset='training',
    seed=123,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size= 32,
    image_size=(224, 224), # changing image size to 224x224
    validation_split=0.2, # 20% for validation
    subset='validation',
    seed=123
)

# Applying preprocess
train_ds = train_ds.map(preprocess_data)
val_ds = val_ds.map(preprocess_data)

# Cache and prefetch to optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).map(
    lambda x, y: (data_augmentation(x), y), 
    num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Define the top layers for classification
def create_model():
    # Base model
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3) # Image size and shape
    )
    
    # Freezing initial layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Building model
    inputs = Input(shape=(224, 224, 3)) # Image size and shape
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Dense layers with dropout
    x = Dense(1024, activation='relu')(x) # Dense layer
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(10, activation='softmax')(x) # Output layer for 10 classes
    
    # Combine base model and top layers into a new model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Creating the object model
model = create_model()

# Compiling
initial_learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        min_delta=0.001
    )
]

# Training the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs= 40,
    callbacks=callbacks,
    verbose=1
)

# Saving the model
model.save('PC-Parts_Classificator_model.keras')

# Plotting training history
plt.figure(figsize=(15, 5))

# Acuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate model
print("\nFinal evaluation of the model:")
final_loss, final_accuracy = model.evaluate(val_ds)
print(f"Final acuracy: {final_accuracy:.4f}")

# Prediction for the confusion matrix
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Confusion matrix
plt.figure(figsize=(12, 10))
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Acuracy by class
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
print("\nAcuracy by class:")
for class_name, accuracy in zip(class_names, class_accuracy):
    print(f'{class_name}: {accuracy:.2%}')