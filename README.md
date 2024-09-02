

---

# **BHAVGEET: An Emotion-Based Music Recommender System 🎵**

---

BHAVGEET is a machine learning-powered music recommendation system that detects the user's emotions in real time through facial and hand gestures and suggests songs based on the identified emotions. It provides a personalized listening experience by selecting music that aligns with your current mood.

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## **Introduction**
BHAVGEET leverages computer vision and deep learning models to detect a user's emotions using facial and hand landmarks captured via a webcam. It then suggests relevant music based on the detected emotion, enhancing the user's listening experience.

## **Features**
- Real-time emotion detection using facial and hand gestures.
- Automatic song recommendations based on emotions, language, and preferred singer.
- Integration with YouTube to play songs that match your mood.

## **How It Works**
1. **Emotion Detection**: The system uses MediaPipe to capture and analyze facial and hand landmarks from the user's webcam feed.
2. **Emotion Classification**: A pre-trained neural network model classifies the detected landmarks into emotions.
3. **Music Recommendation**: The system recommends music from YouTube based on the detected emotion, user-input language, and singer preferences.

## **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/BHAVGEET.git
   cd BHAVGEET
   ```

2. **Install the Required Packages**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Model and Labels**
   - Place the `model.h5` and `labels.npy` files in the project directory.

## **Usage**

1. **Run the Application**
   ```bash
   streamlit run music.py
   ```

2. **Provide Inputs**
   - Enter your preferred language and singer.
   - Let the system capture your emotions through the webcam.

3. **Receive Recommendations**
   - Click the "Recommend me songs" button to get music suggestions based on your detected emotions.

## **File Descriptions**

1. **`music.py`**: The main application file that runs the Streamlit interface, captures emotions, and recommends music.
2. **`data_collection.py`**: Script for collecting facial and hand landmarks data to build the training dataset.
3. **`data_training.py`**: Code for training the neural network model using collected data.
4. **`inference.py`**: A separate script for testing the trained model's emotion detection capabilities.

## **Future Enhancements**
- Expand the emotion dataset to improve classification accuracy.
- Integrate with other music streaming platforms like Spotify or Apple Music.
- Enhance the model to recognize more complex emotions and combinations.
- Implement personalized playlists based on past emotional states.

## **Contributing**
Contributions are welcome! Please fork this repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.



---

Enjoy personalized music recommendations that match your mood with BHAVGEET! 🎶✨



