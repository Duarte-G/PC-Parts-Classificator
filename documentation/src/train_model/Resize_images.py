from PIL import Image
import os

# Directory containing the images
current_dir = os.path.dirname(__file__)  # Script directory
dataset_dir = os.path.join(current_dir, 'dataset')  # Replace with the correct image folder path

# Path to save the resized images (optional)
resized_dir = os.path.join(current_dir, 'dataset_resized')
if not os.path.exists(resized_dir):
    os.makedirs(resized_dir)

# Desired image size
new_size = (256, 256)

# Function to resize all images
def resize_images(input_dir, output_dir, size):
    # Iterate over all folders and files inside the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):  # Filtra por formatos de imagem
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                
                # Resize the image
                img_resized = img.resize(size)
                
                # Path of the new file (keeps the original structure)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # Save the resized image
                img_resized.save(os.path.join(output_path, file))
                print(f'Redimensionado: {file}')

# Run the function
resize_images(dataset_dir, resized_dir, new_size)
