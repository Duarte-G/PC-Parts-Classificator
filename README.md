# Computer Parts Classification

This project uses deep learning to identify computer components from images, including CPUs, GPUs, motherboards, RAM, and more. Leveraging a ResNet50V2 architecture fine-tuned for enhanced accuracy, the model achieves high classification performance across multiple parts. This repository contains the full setup, code structure, and instructions for replicating results, along with insights into model improvements using data augmentation and fine-tuning.
<p align="center">
  <img src="https://github.com/user-attachments/assets/4e2bc8d6-b414-4ee3-9bce-3c3f75876bb4">
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
### Accuracy
The model achieves 90% accuracy on the test set.

### Confusion Matrix
Below is a sample confusion matrix for the classifications:
<p align="center">
  <img src="https://github.com/user-attachments/assets/dd0845ae-e385-4fbc-8820-6ec1d99f09f3">
</p>

