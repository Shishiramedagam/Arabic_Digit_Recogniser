#  Arabic Digit Recognizer

A deep learning-based system that uses a Convolutional Neural Network (CNN) to recognize handwritten Arabic digits (0–9). Built using TensorFlow and Keras, the model achieves over 96% accuracy on the test dataset. This project is ideal for OCR tasks, educational tools, or integrating into applications that require Arabic numeral digitization.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🔍 Overview

Arabic Digit Recognizer is a deep learning project that classifies handwritten Arabic numerals using a CNN. It is trained on a variation of the MNIST dataset tailored for Arabic digits. The goal is to enable digit recognition in documents, forms, or any image-based input.

---

##  Features

- Image-based Arabic numeral recognition (0–9)
- CNN model trained with TensorFlow/Keras
- 96%+ test accuracy
- Prediction script for single images
- Optional Streamlit app for UI-based demo
- Lightweight and fast inference

---

##  Tech Stack

- **Language**: Python 3.10  
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib, Streamlit  
- **Tools**: Jupyter Notebook, VS Code  
- **Framework**: CNN for image classification  

---

##  Dataset

We use the **Arabic Handwritten Digits Dataset** from Kaggle:

- 60,000 training samples  
- 10,000 testing samples  
- Each image is 28×28 pixels, grayscale  
- Labels range from 0 to 9 (in Arabic numerals)

**Source**: [Arabic Digits Dataset on Kaggle](https://www.kaggle.com/datasets/mloey1/ahdd1)

---

##  Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/arabic-digit-recognizer.git
cd arabic-digit-recognizer
