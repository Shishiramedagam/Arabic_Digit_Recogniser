#  Arabic Digit Recognizer using CNN on MNIST Dataset

A deep learning-based system that classifies handwritten Arabic digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project achieves 96% test accuracy and is optimized for lightweight deployment using Flask or Streamlit.

---

##  Abstract

This project presents the design and implementation of a CNN model to recognize handwritten Arabic numerals using a modified version of the MNIST dataset. The solution is designed for high accuracy, portability, and extensibility into real-time applications. It includes complete documentation and logs for reproducibility.

---

##  Tech Stack

- **Language**: Python 3.x  
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib  
- **Deployment (optional)**: Flask / Streamlit  
- **Documentation**: Microsoft Word (IEEE format), Grammarly

---

##  Dataset

- **Name**: Arabic Handwritten Digits Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mloey1/ahdd1)  
- **Structure**:
  - 60,000 training images  
  - 10,000 test images  
  - 28×28 pixel grayscale images  
  - Labels: digits 0 to 9 (in Arabic numerals)

---

##  Model Architecture

The CNN model consists of:

- 2 Convolutional layers (32 and 64 filters)  
- 2 MaxPooling layers (2×2)  
- Dropout layers (0.25 and 0.5)  
- Flatten layer  
- Dense layer with 128 ReLU units  
- Output Dense layer with 10 Softmax neurons  

- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy

---

##  Training and Results

- **Epochs**: 10  
- **Batch Size**: 128  
- **Training Accuracy**: 98.5%  
- **Test Accuracy**: 96.2%  
- Regularization using dropout minimized overfitting  
- Training and validation curves showed strong convergence

---

##  Deployment

The trained model is saved as a `.h5` file and can be deployed using:

- **Flask API** (for backend prediction)  
- **Streamlit UI** (for interactive digit drawing or image upload)  

Supports:
- Real-time digit prediction  
- Static image input  
- REST API response in JSON format

---

##  Project Structure

```text
arabic-digit-recognizer/
├── data/                  # Dataset files
├── models/                # Saved model (.h5)
├── notebooks/             # Jupyter notebooks for EDA and model training
├── train_model.py         # Training script
├── predict.py             # Prediction CLI script
├── app.py                 # Streamlit or Flask app (optional)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
