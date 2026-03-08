
---

# Sign Language Detection

A **deep learning based sign language recognition system** that detects **American Sign Language (ASL) alphabet gestures** using a webcam.

The system uses a **Convolutional Neural Network** implemented with **Keras** and **TensorFlow**, while image capture and preprocessing are handled using **OpenCV**.

The model recognizes **hand gestures representing English alphabets (A–Z)** and predicts them in real time using a webcam.

Repository:
[https://github.com/Abineshabee/Sing-Language-Detection.git](https://github.com/Abineshabee/Sing-Language-Detection.git)

---

# Project Overview

Communication with people who use **sign language** can be difficult without interpreters.

This project demonstrates how **computer vision and deep learning** can help bridge that gap by automatically recognizing hand gestures from video input.

The system works in three stages:

1. Dataset creation using webcam
2. CNN model training
3. Real-time gesture recognition

---

# Project Architecture

```text
Webcam Input
     ↓
Hand Segmentation (HSV Mask)
     ↓
Image Preprocessing (64x64)
     ↓
CNN Model
     ↓
Alphabet Prediction (A–Z)
```

---

# Example Output

The model predicts hand gestures corresponding to **American Sign Language alphabets**.

Example confusion matrix after training:

Most predictions lie along the **diagonal**, which indicates high classification accuracy.

---

# Technologies Used

| Technology | Purpose                         |
| ---------- | ------------------------------- |
| Python     | Main programming language       |
| OpenCV     | Image capture and preprocessing |
| TensorFlow | Deep learning backend           |
| Keras      | CNN model implementation        |
| NumPy      | Numerical operations            |

---

# Features

• Real-time gesture recognition using webcam
• Custom dataset creation
• HSV mask-based hand segmentation
• CNN-based classification
• Support for A–Z sign language alphabets
• Adjustable segmentation using HSV trackbars

---

# Dataset Creation

The dataset is captured using **OpenCV webcam input**.

The system extracts the hand region and converts it into a **binary mask** using HSV color segmentation.

### Hand Segmentation Pipeline

```text
Webcam Frame
     ↓
Region of Interest (ROI)
     ↓
Convert to HSV
     ↓
Apply Color Mask
     ↓
Binary Hand Image
     ↓
Resize to 64x64
     ↓
Save as Dataset Image
```

---

# Dataset Structure

```text
mydata
│
├── training_set
│   ├── A
│   ├── B
│   ├── C
│   └── ...
│
└── test_set
    ├── A
    ├── B
    ├── C
    └── ...
```

Each folder contains images corresponding to one alphabet gesture.

---

# Mask-Based Hand Detection

The system uses **HSV color thresholding** to isolate the hand from the background.

HSV ranges can be adjusted using trackbars.

Example parameters:

```python
lower_blue = np.array([l_h, l_s, l_v])
upper_blue = np.array([u_h, u_s, u_v])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
```

This produces a **binary mask of the hand region**.

---

# Dataset Capture Script

The dataset is generated using:

```bash
python capture.py
```

Steps:

1. Enter gesture name (A, B, C, etc)
2. Adjust HSV trackbars for proper hand segmentation
3. Press **C** to capture images

Dataset size:

| Dataset      | Images     |
| ------------ | ---------- |
| Training Set | 350 images |
| Test Set     | 50 images  |

All images are resized to:

```
64 x 64 pixels
```

---

# CNN Model

The classifier uses a **Convolutional Neural Network** architecture.

Typical CNN layers:

```text
Input Layer (64x64 image)
      ↓
Convolution Layer
      ↓
ReLU Activation
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense Layer
      ↓
Softmax Output (A-Z)
```

---

# Training the Model

Run:

```bash
python cnn_model.py
```

This script:

• Loads training images
• Preprocesses them
• Trains the CNN model
• Saves the trained model

---

# Real-Time Gesture Recognition

After training the model, run:

```bash
python recognise.py
```

The program will:

1. Open the webcam
2. Detect the hand using HSV mask
3. Feed the image to the CNN
4. Display predicted alphabet

---

# Example Recognition Flow

```text
User shows hand gesture
      ↓
Webcam captures frame
      ↓
Hand segmentation
      ↓
Resize to 64x64
      ↓
CNN prediction
      ↓
Display predicted alphabet
```

---

# Installation

Install required libraries:

```bash
pip install opencv-python tensorflow keras numpy
```

---

# How to Run the Project

Step 1 – Create dataset

```bash
python capture.py
```

Step 2 – Train the model

```bash
python cnn_model.py
```

Step 3 – Run gesture recognition

```bash
python recognise.py
```

---

# Applications

This system can be used in:

• Assistive communication systems
• Sign language translation tools
• Educational tools for learning sign language
• Accessibility software
• Human-computer interaction research

---

# Future Improvements

Possible enhancements:

• Support for full sign language words
• Real-time sentence generation
• Deep learning model optimization
• Mobile deployment
• Integration with speech synthesis

---

# Author

Abinesh N

GitHub
[https://github.com/Abineshabee](https://github.com/Abineshabee)

---

