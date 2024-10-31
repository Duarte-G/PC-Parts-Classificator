# Computer Parts Classification

This project uses deep learning to identify computer components from images, including CPUs, GPUs, motherboards, RAM, and more. Leveraging a ResNet50V2 architecture fine-tuned for enhanced accuracy, the model achieves high classification performance across multiple parts. This repository contains the full setup, code structure, and instructions for replicating results, along with insights into model improvements using data augmentation and fine-tuning.
<p align="center">
  <img src="https://github.com/user-attachments/assets/25cd6e60-f697-4c41-a737-c76fb371c841">
</p>

## About the Project
This project aims to classify images of computer parts. The model is currently trained to recognize the following classes:
- CPU
- GPU
- Motherboard
- Computer Case
- HDD
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

