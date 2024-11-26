# Emotion Recognition Using CNN

This project focuses on building a Convolutional Neural Network (CNN) model to recognize human emotions from facial images. The system classifies emotions into five categories: `BinhThuong`, `Buon`, `GianDu`, `NgacNhien`, and `VuiVe`.

## Features

- **Dataset Preparation**: Extracted 6000 images from each labeled video (`BinhThuong`, `Buon`, `GianDu`, `NgacNhien`, `VuiVe`).
- **CNN Model Architecture**: A deep learning model using Keras and TensorFlow frameworks for emotion classification.
- **Training & Evaluation**: Trained on a custom image dataset with data augmentation techniques to improve performance.
- **Optimization Techniques**: Implemented methods like Dropout and Momentum to enhance convergence and reduce overfitting.

## Dataset

The dataset consists of:
- 5 folder image labeled as `BinhThuong`, `Buon`, `GianDu`, `NgacNhien`, and `VuiVe`.
- Images are organized into folders corresponding to their emotion labels.

## Requirements

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Tkinter
- Customtkinter

Install the required libraries using:
```bash
pip install -r requirements.txt
