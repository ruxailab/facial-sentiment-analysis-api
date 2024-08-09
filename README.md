# RuxaiLab Facial Emotion Detection API

## Overview
This Flask API serves as the backend for the RuxaiLab Bachelor's thesis project, providing facial emotion detection capabilities. It processes video input, detects emotions displayed on faces within the video frames, and integrates with Firebase.

## Features
- **Video Processing:** Upload a video file to analyze emotions displayed on faces within the frames.
- **Facial Emotion Detection:** Utilizes a pre-trained model to analyze emotions in video frames.
- **Firebase Integration:** User authentication and storage are managed through Firebase, ensuring secure access to the API.
- **Logging:** Utilizes Python's logging module for informational and error messages.

## Setup
1. Clone the repository: `git clone https://github.com/your-username/ruxailab-facial-emotion-api.git`
2. Install Poetry (if not already installed): [Poetry Installation Guide](https://python-poetry.org/docs/#installation)
3. Install dependencies: `poetry install`
4. Configure Firebase:
5. Start the API: `poetry run python app.py`

## Usage
### Uploading a Video
- Send a POST request to `/process_video` endpoint with the name of the video file to be analyzed in the request body.
- The API will download the video from Firebase Storage, perform emotion analysis, and store the results in Firestore.
- Once the API is running, it automatically detects new videos uploaded to Firebase Storage, processes them to analyze emotions, and uploads the results to Firestore.

### Testing Firebase Function
- Send a GET request to `/test` endpoint.
- The API will call a Firebase function hosted elsewhere for testing purposes.

## API Routes
- **POST /process_video:** Initiates emotion analysis on the uploaded video.
- **GET /test:** Calls a Firebase function for testing purposes.

