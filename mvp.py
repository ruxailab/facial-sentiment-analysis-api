import cv2
from keras.models import model_from_json
import numpy as np
import tensorflow as tf

#This file is initial implementation as minimum viable product for emotion detection needed for the project agile development

def load_model():
   
    model = tf.keras.models.load_model("models/model2/model2.h5")
    return model

def load_face_cascade():
   
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    return face_cascade

def extract_features(image):
   
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    feature = feature / 255.0
    return feature

def predict_emotion(model, img):
  
    pred = model.predict(img)
    return pred

def getPercentages(predictions):
    
    percentages = []
    emotionCountMap = {
        'Angry': 0,
        'Disgusted': 0,
        'Fearful': 0,
        'Happy': 0,
        'Neutral': 0,
        'Sad': 0,
        'Surprised': 0
    }

    for prediction in predictions:
        if prediction == 'Angry':
            emotionCountMap["Angry"] += 1
        elif prediction == 'Disgusted':
            emotionCountMap["Disgusted"] += 1
        elif prediction == 'Fearful':
            emotionCountMap["Fearful"] += 1
        elif prediction == 'Happy':
            emotionCountMap["Happy"] += 1
        elif prediction == 'Neutral':
            emotionCountMap["Neutral"] += 1
        elif prediction == 'Sad':
            emotionCountMap["Sad"] += 1
        elif prediction == 'Surprised':    
            emotionCountMap["Surprised"] += 1
    percentages = {
        'Angry': round((emotionCountMap["Angry"] / len(predictions) * 100), 2),
        'Disgusted': round((emotionCountMap["Disgusted"] / len(predictions) * 100), 2),
        'Fearful': round((emotionCountMap["Fearful"] / len(predictions) * 100), 2),
        'Happy': round((emotionCountMap["Happy"] / len(predictions) * 100), 2),
        'Neutral': round((emotionCountMap["Neutral"] / len(predictions) * 100), 2),
        'Sad': round((emotionCountMap["Sad"] / len(predictions) * 100), 2),
        'Surprised': round((emotionCountMap["Surprised"] / len(predictions) * 100), 2)
        
    }
    print(emotionCountMap)
    print(percentages)    

def main(video_path, predictions):

    model = load_model()
    face_cascade = load_face_cascade()

    # pass in video_path or 0 for webcam
    video = cv2.VideoCapture("fv.mp4")

    # Define emotion labels
    labels = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}
    counter = 0
    while True:
        # Read a frame from video
        ret, im = video.read()
        counter += 1
        if not ret:
            break

        # Convert the frame to gray color
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(im, 1.3, 5)

        try:
            for (p, q, r, s) in faces:
                # Extract region of interest
                image = gray[q:q + s, p:p + r]
                # Draw rectangle around the face
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                # Resize the image
                image = cv2.resize(image, (48, 48))
                # Extract features
                img = extract_features(image)
                # Make prediction
                pred = predict_emotion(model, img)
                # Get the index of the highest probability
                prediction_label = labels[pred.argmax()]

                predictions.append(prediction_label)
                # Display the emotion
                cv2.putText(im, prediction_label, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # Display the frame
            cv2.imshow('Emotion Detector', im)
            # Break the loop if 'q' is pressed
            print(counter)
            if cv2.waitKey(1) == 27:
                break

        except cv2.error:
            pass

    # Release the video
    video.release()
    # Close the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predictions = []
    main("static/videos/face_video.mp4",predictions)
    print(predictions)
    getPercentages(predictions)
