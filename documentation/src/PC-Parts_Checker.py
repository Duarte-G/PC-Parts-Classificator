import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

class PCPartsChecker:
    def __init__(self, model_path='ResNet50V2_model.keras'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = 224

        #  Components needed to build a PC
        self.required_components = {
            'case': 'Computer Case',
            'cpu': 'CPU',
            'gpu': 'Graphics Card',
            'hdd': 'HD/SSD',
            'headset': 'Headset',
            'keyboard' : 'Keyboard',
            'monitor' : 'Monitor',
            'motherboard': 'Motherboard',
            'mouse' : 'Mouse',
            'ram': 'RAM Memory'
        }

        # List of components already added
        self.added_components = set()

    def preprocess_image(self, image_path):
        # Processes the image for classification
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
        return img_array

    def classify_component(self, image_path):
        # Classifies a component from an image
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        predicted_class = list(self.required_components.keys())[predicted_class_index]
        return predicted_class, confidence

    def add_component(self, image_path):
        # Adds a component to the build
        component_type, confidence = self.classify_component(image_path)
        if confidence < 0.6:
            return f"Low confidence: {confidence:.2%}. Try again with another image."
        
        self.added_components.add(component_type)
        return f"{self.required_components[component_type]} successfully added!"

def main_menu(checker):
    while True:
        print("\nChoose an option:")
        print("1 - Add a component")
        print("2 - Log out")

        choice = input("Option: ")
        if choice == '1':
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image", "*.png *.jpg *.jpeg")]
            )
            if file_path:
                result = checker.add_component(file_path)
                print(result)
        elif choice == '2':
            print("Logging out...")
            break
        else:
            print("Invalid option, try again.")

if __name__ == "__main__":
    checker = PCPartsChecker()
    main_menu(checker)
