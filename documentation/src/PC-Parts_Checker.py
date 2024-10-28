import tensorflow as tf
import numpy as np
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import json

class PCPartsChecker:
    def __init__(self, model_path='ResNet50V2_model.keras'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = 224

        #  Components needed to build a PC
        self.required_components = {
            'case': {'required': True, 'name': 'Computer Case', 'description': 'Protects and organizes components'},
            'cpu': {'required': True, 'name': 'CPU', 'description': 'Processes all information'},
            'gpu': {'required': True, 'name': 'Graphics Card', 'description': 'For games or graphic work. Optional if your CPU has an integrated graphic card'},
            'hdd': {'required': True, 'name': 'HD/SSD', 'description': 'Data storage'},
            'headset': {'required': False, 'name': 'Headset', 'description': 'Audio and communication'},
            'keyboard': {'required': True, 'name': 'Keyboard', 'description': 'Entering text and commands'},
            'monitor': {'required': True, 'name': 'Monitor', 'description': 'Display'},
            'motherboard': {'required': True, 'name': 'Motherboard', 'description': 'Connects all components'},
            'mouse': {'required': True, 'name': 'Mouse', 'description': 'Cursor control'},
            'ram': {'required': True, 'name': 'RAM Memory', 'description': 'Quick access memory'},
        }

        # List of components already added
        self.added_components = set()

        # Load class_names
        self.class_names = sorted(list(self.required_components.keys()))
        
        # Create directory to save component images
        self.save_dir = Path('pc_build_components')
        self.save_dir.mkdir(exist_ok=True)
        
        # Load save components
        self.load_saved_components()

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
        try:
            component_type, confidence = self.classify_component(image_path)
            # Check minimum confidence
            if confidence < 0.6:
                return {
                    'success': False,
                    'message': f'It was not possible to identify the component with certainty (confidence: {confidence:.2%})'
                }
            
            # Copy image to build directory
            new_image_path = self.save_dir / f"{component_type}_{len(self.added_components)}.jpg"
            Image.open(image_path).save(new_image_path)
   
            # Add to component list
            self.added_components.add(component_type)
            # Save current status
            self.save_current_state()
            
            return {
                'success': True,
                'message': f'Componente {self.required_components[component_type]["name"]} successfully added! (trust: {confidence:.2%})'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error when adding component: {str(e)}'
            }
    def get_missing_components(self):
        # Returns list of missing components and recommendations
        missing = []
        optional = []
        
        for component, info in self.required_components.items():
            if component not in self.added_components:
                if info['required']:
                    missing.append({
                        'type': component,
                        'name': info['name'],
                        'description': info['description']
                    })
                else:
                    optional.append({
                        'type': component,
                        'name': info['name'],
                        'description': info['description']
                    })
        
        return {
            'missing': missing,
            'optional': optional,
            'is_complete': len(missing) == 0
        }

    def save_current_state(self):
        # Saves the current state of the components
        state = {
            'components': list(self.added_components)
        }
        with open(self.save_dir / 'build_state.json', 'w') as f:
            json.dump(state, f)

    def load_saved_components(self):
        # Load saved components
        state_file = self.save_dir / 'build_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.added_components = set(state['components'])

    def reset_build(self):
        """Reseta a build atual"""
        self.added_components.clear()
        for file in self.save_dir.glob('*.*'):
            if file.name != '.gitkeep':  # Preservar arquivo .gitkeep se existir
                file.unlink()
        self.save_current_state()
        return "Build sucessfully reset!"

def select_image_file():
    root = tk.Tk()
    root.withdraw() # Hides the main Tkinter window
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select an image",
        filetypes=[("Image", "*.png *.jpg *.jpeg")]
    )
    root.destroy()  # Closes the Tkinter instance after selection
    return file_path

def main_menu(checker):
    while True:
        print("\nChoose an option:")
        print("1 - Add a component")
        print("2 - Check missing components")
        print("3 - Reset build")
        print("4 - Exit")

        choice = input("Option: ")
        if choice == '1':
            file_path = select_image_file() # Calls the function to select the file
            if file_path:
                result = checker.add_component(file_path)
                print(result['message'])
            else:
                print("No files have been selected.")
        elif choice == '2':
            missing_info = checker.get_missing_components()
            if missing_info['is_complete']:
                print("Build is complete!")
            else:
                print("Missing components:")
                for comp in missing_info['missing']:
                    print(f"- {comp['name']}: {comp['description']}")
                print("Optional components:")
                for comp in missing_info['optional']:
                    print(f"- {comp['name']}: {comp['description']}")
        elif choice == '3':
            print(checker.reset_build())
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option, try again.")

if __name__ == "__main__":
    checker = PCPartsChecker()
    main_menu(checker)
