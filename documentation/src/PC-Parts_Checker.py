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
        # Path to the current script folder
        current_dir = os.path.dirname(__file__)
        full_model_path = os.path.join(current_dir, model_path)
        # Load the trained model
        self.model = tf.keras.models.load_model(full_model_path)
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
        # Reset build
        self.added_components.clear()
        for file in self.save_dir.glob('*.*'):
            if file.name != '.gitkeep': # Maintain gitkeep file
                file.unlink()
        self.save_current_state()
        return "Build sucessfully reset!"

class PCPartsGUI:
    def __init__(self):
        self.checker = PCPartsChecker()
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("PC-Parts Verificator")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main frame with background style
        main_frame = tk.Frame(self.root, padx=20, pady=20, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Button Style
        button_style = {"font": ("Arial", 12, "bold"), "bg": "#4CAF50", "fg": "white", "relief": "groove", "bd": 3, "width": 25}
        tk.Button(main_frame, text="Add component", command=self.add_component, **button_style).pack(pady=10)
        tk.Button(main_frame, text="Check for missing components", command=self.check_missing, **button_style).pack(pady=10)
        tk.Button(main_frame, text="Reset Build", command=self.reset_build, **button_style).pack(pady=10)

        # Frame for list of components with border
        self.components_frame = tk.LabelFrame(main_frame, text="Added components", padx=10, pady=10, bg="#ffffff", font=("Arial", 12, "bold"), fg="#333")
        self.components_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Update components list
        self.update_components_list()

    def on_closing(self):
        # Called when the window is closed
        if messagebox.askyesno("Confirm exit", "Do you want to reset the build before you leave?"):
            self.checker.reset_build()
        self.root.destroy()

    def add_component(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg")]
        )
        if file_path:
            result = self.checker.add_component(file_path)
            messagebox.showinfo("Results", result['message'])
            self.update_components_list()

    def check_missing(self):
        result = self.checker.get_missing_components()
        
        message = "Build status:\n\n"
        
        if result['is_complete']:
            message += "✅ Your build is complete with all the necessary components!\n\n"
        else:
            message += "❌ Some necessary components are still missing:\n\n"
        
        if result['missing']:
            message += "Necessary components missing:\n"
            for comp in result['missing']:
                message += f"- {comp['name']}: {comp['description']}\n"
            message += "\n"
        
        if result['optional']:
            message += "Optional components you can add:\n"
            for comp in result['optional']:
                message += f"- {comp['name']}: {comp['description']}\n"
        
        messagebox.showinfo("Missing components", message)

    def reset_build(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the current build?"):
            message = self.checker.reset_build()
            messagebox.showinfo("Reset", message)
            self.update_components_list()

    def update_components_list(self):
        # Clear current list
        for widget in self.components_frame.winfo_children():
            widget.destroy()

        # List added components
        for component in self.checker.added_components:
            name = self.checker.required_components[component]['name']
            tk.Label(self.components_frame, text=f"✓ {name}", font=("Arial", 10), bg="#ffffff", fg="#333").pack(anchor='w', padx=5, pady=2)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PCPartsGUI()
    app.run()