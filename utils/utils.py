import cv2
import numpy as np
import tensorflow as tf
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import shutil
from utils.logger import get_logger

logger = get_logger(__name__)


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

def delete_video():
    """Remove all files and subdirectories under static/videos/."""
    folder = "static/videos/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            logger.debug("Deleted: %s", file_path)
        except Exception as e:
            logger.error("Failed to delete %s: %s", file_path, e)

def split_video_into_clips(video_path):
    """Split a video into 15-second clips and move them to static/videos/."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = frame_count / fps
    cap.release()

    logger.info("Video length: %d seconds", int(seconds))

    video_paths = []
    for i in range(0, int(seconds), 10):
        starttime = i
        endtime = i + 15
        targetname = str(i) + ".mp4"
        video_paths.append('static/videos/' + targetname)
        ffmpeg_extract_subclip(video_path, starttime, endtime, targetname=targetname)
        logger.debug("Moving clip %s to static/videos/", targetname)
        logger.debug("Absolute path: %s", os.path.abspath(targetname))
        shutil.move(targetname, 'static/videos/')

    return video_paths
