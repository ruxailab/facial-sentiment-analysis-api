import cv2
import numpy as np
import tensorflow as tf
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import shutil
    

def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

def load_face_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1) / 255.0
    return feature

def predict_emotion(model, img):
    return model.predict(img)

def getPercentages(predictions):
    emotion_count_map = {emotion: 0 for emotion in ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']}
    for prediction in predictions:
        emotion_count_map[prediction] += 1
    if len(predictions) == 0:
        percentages = {emotion: 0 for emotion in emotion_count_map.keys()}
    else:
        percentages = {emotion: round((count / len(predictions) * 100), 2) for emotion, count in emotion_count_map.items()}
    return percentages

#Delete /static/videos/ content
def delete_video():
    folder = "static/videos/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#Split videos
def split_video_into_clips(video_path):
    #Get video length
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #we get the total number of frames of the video
    seconds = frame_count/fps
    print(f'Video length: {int(seconds)} seconds')
    cap.release()
    video_paths = []
    #Split video into clips of 10 seconds
    for i in range(0, int(seconds), 10):
        starttime = i
        endtime = i+15 
        targetname = str(i)+".mp4"
        video_paths.append('static/videos/'+targetname)
        ffmpeg_extract_subclip(video_path, starttime, endtime, targetname=targetname)
        #move the video to the clips folder
        print(f"Moving {targetname} to clips folder")
        #show acutal path
        print(os.path.abspath(targetname))
        shutil.move(targetname, 'static/videos/')
    return video_paths
