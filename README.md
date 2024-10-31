# Computer Parts Classification

This project uses deep learning to identify computer components from images, including CPUs, GPUs, motherboards, RAM and more. Using the ResNet50V2 architecture tuned for greater accuracy, the model achieved good multi-part classification performance. This repository contains the full setup, code structure, and instructions for replicating results, along with insights into model improvements using data augmentation and fine-tuning.
<p align="center">
  <img src="https://github.com/user-attachments/assets/25cd6e60-f697-4c41-a737-c76fb371c841">
</p>

## Download and Installation
### Use the Checker
Follow these steps to run the program on your own machine:
#### 1. Clone the Repository:
```python
git clone https://github.com/Duarte-G/PC-Parts-Classificator.git
cd PC-Parts-Classificator
cd documentation
cd src
```

#### 2. Install Required Packages: Make sure you have Python 3.7 or later, and install the dependencies.
```python
pip install -r requirements.txt
```
#### 3. Run the Program: Run the GUI to start the classification
```python
python PC-Parts_Checker.py
```

### Customizing the Model and Dataset
If you'd like to experiment with the model architecture, fine-tune it with additional data, or even adjust the training parameters, follow these steps:
#### 1. Modify the Model Architecture
The model architecture is defined in `train_model/PC-Parts_Classificator.py`. You can open this file and adjust layers, add regularization, or swap out ResNet50V2 for another model.
```python
# Example: Switch to a different base model
from tensorflow.keras.applications import VGG16

    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3) # Image size and shape
    )
```
#### 2. Add More Data to the Dataset
You can expand the dataset by adding more images in the folders (```dataset/case/, dataset/gpu/, ...```). Ensure the new images are correctly labeled and follow the same format as the existing dataset (```256x256```).

If the images are in different resolutions, you can use the provided code to automatically resize all images in a folder to the desired resolution. Before running, edit the code to specify the folder name containing the images you want to resize, then add the images to the dataset. This will ensure consistency across the dataset:
```python
dataset_dir = os.path.join(current_dir, 'dataset')  # Replace with the correct image folder path
```
```python
python train_model/Resize_images.py
```
Then, retrain the model to incorporate the new data.
```python
python train_model/PC-Parts_Classificator.py
```
#### 3. Test the Updated Model
Finally, evaluate the new model’s performance on your test set or with real images:
```bash
python PC-Parts_Checker.py
```

## About the Project
This project aims to classify images of computer parts. The model is currently trained to recognize the following classes:
- CPU
- GPU
- Motherboard
- Computer Case
- HDD/SSD
- RAM
- Headset
- Monitor
- Mouse
- Keyboard

The approach involves a convolutional neural network (ResNet50V2) fine-tuned with data augmentation techniques to improve robustness and accuracy.

## Technologies Used
- Language: Python
- Environment: Visual Studio Code
- Model: ResNet50V2
- Framework: TensorFlow / Keras

## Sample Results

### Confusion Matrix
Below is a sample confusion matrix for the classifications:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8c45ea01-5478-408f-bbb7-2972d47daa06">
</p>

## Model and Classfication Overview
The classification model for this project is built on a ResNet50V2 architecture, a great CNN for image recognition tasks. It was very useful for distinguishing visually similar computer components, such as GPUs, motherboards, and RAM.

### Model Structure and Fine-Tuning
The model's structure includes:
- Layer Freezing: The main layers were frozen to maintain the essential visual features of ImageNet, while the deeper layers were trained specifically for this project.
- Batch Normalization: Applied to enhance model stability and speed up training.

### Data Augmentation
To enhance model robustness, data augmentation techniques were used during training, including:
- Horizontal Flipping
- Small Rotations (10%)
- Zoom Adjustments (10%)
- Minor Translations (10%)

These augmentations contributed to the model's performance in recognizing components across a range of real-world image variations.

## Future Improvements
Some potential improvements, such as:
- Expanding the dataset to include additional components.
- Adding more comprehensive error handling for diverse image quality or backgrounds.

