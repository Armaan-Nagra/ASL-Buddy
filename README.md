# ASL Interpreter & Quiz App

![HackSheffield 9 Winner](https://img.shields.io/badge/HackSheffield%209-Winner-brightgreen)
![Frasers Group Sponsor Prize](https://img.shields.io/badge/Frasers%20Group-Sponsor%20Prize-blue)
![SWICS Sponsor Prize](https://img.shields.io/badge/SWICS-Award-orange)

An award-winning iOS application that leverages cutting-edge computer vision to interpret American Sign Language (ASL) and gamifies learning through interactive quizzes. This project won the **Top Sponsor Prize at HackSheffield 9** (sponsored by Frasers Group) and an additional prize from **SWICS**.

## 📸 See It in Action

Check out our project submission on **Devpost** to see more details, screenshots, and a demo of the app:

[Visit Our Devpost Submission](https://devpost.com/software/bsl-buddy)

## 🚀 Features

- **Real-Time ASL Recognition:** Uses computer vision to detect and interpret ASL gestures.
- **Interactive Quizzes:** Tests and reinforces the user's ASL skills with engaging quizzes.
- **Educational Experience:** Helps users learn ASL in an accessible and interactive manner.
- **Seamless iOS Integration:** Designed specifically for iOS devices for a native experience.

## 🏆 Awards

- **Top Sponsor Prize** - Frasers Group at HackSheffield 9
- **SWICS Prize**

## 🔧 Technologies Used

### Final Implementation
- **iOS Development**: Built with Swift and SwiftUI to provide a smooth, responsive, and native app interface.
- **Core ML**: Leveraged Apple's Core ML framework for efficient on-device ASL gesture recognition, ensuring real-time performance and seamless integration with iOS.

### First Prototype
- **Computer Vision**: Used `mediapipe` and `opencv` to enable real-time ASL gesture detection.
- **Machine Learning**:
  - Implemented a `Random Forest Classifier` using `scikit-learn` for gesture recognition during early development.
  - Utilized `pickle` for saving and loading the trained model.
- **Data Handling**:
  - Processed gesture data using NumPy to ensure compatibility and efficiency.
  - Applied `train_test_split` from `sklearn` to partition data into training and test sets.
- **Video Capture**: Integrated `cv2.VideoCapture` to enable real-time hand gesture detection in the prototype.
- **Hand Tracking**: Used `mediapipe Hands` for precise hand landmark detection and tracking in the initial testing phase.

