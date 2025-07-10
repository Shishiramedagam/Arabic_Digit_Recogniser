**Title**

**Arabic Digit Recognizer using CNN on MNIST Dataset**


**Author**

**Medagam Shishira Reddy**
Department of Computer Science,
Keshav Memorial Engineering College,
Hyderabad, India


**Abstract**

This paper presents the design and implementation of a Convolutional Neural Network (CNN) to classify handwritten Arabic numerals from the MNIST dataset. The model is trained using TensorFlow and Keras, achieving 96% accuracy on the test set. The project involves preprocessing, model design, evaluation, and documentation. The solution is efficient, lightweight, and suitable for mobile or web deployment. Comprehensive documentation was created using Microsoft Word and technical writing templates.


**Keywords**

Arabic digits, MNIST, CNN, digit recognition, deep learning, Keras, TensorFlow


**1. Introduction**

Automatic recognition of handwritten Arabic numerals is a fundamental task in image processing and pattern recognition. It has applications in digital forms, postal codes, banking systems, and document digitization. In this project, we developed a Convolutional Neural Network (CNN) using Keras and TensorFlow to classify digits from the Arabic subset of the MNIST dataset.


**2. Dataset**

We used a publicly available Arabic Handwritten Digits dataset based on the MNIST format. The dataset consists of 60,000 training images and 10,000 test images, each 28x28 pixels in grayscale, labeled from 0 to 9.



**3. Model Architecture**

The CNN model consists of:

* 2 Convolutional layers (32 and 64 filters)
* 2 MaxPooling layers (2×2)
* Dropout (0.25, 0.5)
* Flatten layer
* Dense layer (128 neurons with ReLU)
* Output Dense layer (10 neurons with Softmax)

The model is compiled with categorical cross-entropy loss, Adam optimizer, and accuracy as the evaluation metric.


 **4. Training and Results**

The model was trained for 10 epochs with a batch size of 128. It achieved:

* **Training Accuracy:** 98.5%
* **Test Accuracy:** 96.2%
* Minimal overfitting due to regularization (dropout)
* Validation curves showed good convergence


**5. Deployment**

The trained model is saved in `.h5` format and can be deployed using a simple Flask or Streamlit interface. The model supports both live-drawing and static image input for predictions.


**6. Documentation**

All components of the project are documented using Microsoft Word in an IEEE-style format. Documentation includes:

* System architecture
* Training logs and accuracy graphs
* Model summary
* Usage instructions
* Deployment guide

Grammarly was used to improve writing quality and consistency.


**7. Conclusion**

This project demonstrates the effectiveness of CNNs for digit recognition using the Arabic MNIST dataset. The system achieves high accuracy with low computational requirements and can be integrated into applications that require Arabic digit recognition.


**References**

1. Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, vol. 86, no. 11, 1998.
2. François Chollet, "Keras: Deep learning library," [https://keras.io](https://keras.io)
3. Abadi et al., "TensorFlow: A system for large-scale machine learning," OSDI 2016.
4. Arabic Handwritten Digits Dataset, [https://www.kaggle.com/datasets/mloey1/ahdd1](https://www.kaggle.com/datasets/mloey1/ahdd1)



