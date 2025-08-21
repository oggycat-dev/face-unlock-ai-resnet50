# 🤖 Face Unlock System

This is an AI project that uses Deep Learning to build a facial authentication system via a computer's camera. This project is part of a learning and research process in Computer Vision and Deep Learning models.

## ✨ Features

- **Enrollment:** Captures and saves a unique "digital fingerprint" (embedding) for a user's face.
- **Verification:** Compares the current face with the registered data to authenticate the user's identity in real-time.

## 🛠️ Tech Stack

- **Language:** Python
- **Core Libraries:** TensorFlow, Keras, OpenCV, Scipy, NumPy
- **Model Architecture:** ResNet50 (trained using Transfer Learning)
- **Training Dataset:** LFW (Labelled Faces in the Wild)

## 🚀 How to Use

1.  **Clone this repository to your machine:**
    ```bash
    git clone [Your-GitHub-Repository-URL]
    cd [your-repository-name]
    ```

2.  **Install the required libraries:**
    ```bash
    pip3 install tensorflow opencv-python scipy numpy
    ```

3.  **Download the Model File:**
    The model file `face_recognition_resnet50.h5` is too large for GitHub. You need to download it manually from the link below and place it in the project directory:
    **[Paste the download link for your .h5 model here]**

4.  **Enroll your face:**
    Open a Terminal/Command Prompt in the project directory and run the command:
    ```bash
    python face_unlock.py --mode enroll --name [YourName]
    ```

5.  **Run the verification mode:**
    After successful enrollment, run the following command to start the unlock system:
    ```bash
    python face_unlock.py --mode verify
    ```

## 🧠 Model Training

The entire ResNet50 model training process was conducted on the Kaggle platform to leverage free GPUs. You can view the details and rerun the entire process here:
**[https://www.kaggle.com/code/tuandatdanh/face-detect]**

---
