# Tamil Sign Language Recognition

## Overview
This project is a **Tamil Sign Language Recognition System** designed to identify 247 unique hand gestures in Tamil sign language with 80% accuracy. The project was developed as part of the **Millennium Fellowship** and aims to bridge communication gaps by leveraging computer vision and machine learning.

## Features
- Recognizes **247 unique Tamil sign language gestures**.
- Uses **OpenCV camera modules** for real-time gesture capture.
- Employs **pose estimation** to extract key hand landmarks.
- Utilizes a **Random Forest Classifier** for gesture prediction.
- Achieves an **80% prediction accuracy**.

## Tech Stack
- **Python**
- **OpenCV**: For capturing and processing video feed.
- **Pose Estimation**: Extracting key hand landmarks.
- **Scikit-learn**: Implementing the Random Forest Classifier.
- **NumPy & Pandas**: Data preprocessing and manipulation.

## How It Works
1. **Real-Time Video Feed**: Captures hand gestures using OpenCV.
2. **Pose Estimation**: Detects and extracts hand landmarks (key points).
3. **Feature Extraction**: Converts the hand landmarks into feature vectors.
4. **Classification**: Random Forest Classifier predicts the corresponding gesture class.
5. **Output**: Displays the recognized gesture as text.



## Dataset
- The dataset consists of 247 gesture classes with corresponding labeled samples.
- Hand landmarks are extracted and saved as feature vectors.

## Results
- **Prediction Accuracy**: 80%
- **Number of Classes**: 247 Tamil Sign Language gestures

## Challenges & Learnings
- **Pose Estimation**: Fine-tuning the detection of hand landmarks.
- **Data Diversity**: Ensuring a variety of samples for each gesture class.
- **Model Optimization**: Achieving a balance between accuracy and computational efficiency.

## Future Scope
- Enhance accuracy with additional training data.
- Expand to include more sign languages.
- Implement real-time deployment on mobile or web platforms.

## Contributors
- **Yeshwanth Balaji A P**
- **Harith**
- **Anudeep**

## Acknowledgments
We are grateful to the **Millennium Fellowship** for providing us with this platform to innovate and make a difference.

## License
This project is licensed under the [MIT License](LICENSE).
