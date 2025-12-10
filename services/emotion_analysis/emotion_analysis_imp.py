import os
from schemas.emotion_schema import GetEmotionPercentagesResponse
from services.emotion_analysis.emotion_analysis_service import EmotionsAnalysisService
import logging
import coloredlogs
from utils.utils import load_model, load_face_cascade, extract_features, predict_emotion, getPercentages
import cv2

class EmotionsAnalysisImp(EmotionsAnalysisService):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.face_cascade = load_face_cascade()
        coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def get_emotion_percentages(self, video_path: str) -> GetEmotionPercentagesResponse:
        predictions = []
        labels = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}
        self.logger.info(f"Loading video from path: {video_path}")

        if not os.path.exists(video_path):
            self.logger.error(f"Video file does not exist: {video_path}")
            directory = os.path.dirname(video_path)
            if os.path.exists(directory):
                self.logger.info(f"Contents of the directory {directory}:")
                for item in os.listdir(directory):
                    self.logger.info(f" - {item}")
            else:
                self.logger.error(f"Directory does not exist: {directory}")
            return GetEmotionPercentagesResponse(Angry=0, Disgusted=0, Fearful=0, Happy=0, Neutral=0, Sad=0, Surprised=0)

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            self.logger.error(f"Failed to open video file: {video_path}")
            return GetEmotionPercentagesResponse(Angry=0, Disgusted=0, Fearful=0, Happy=0, Neutral=0, Sad=0, Surprised=0)

        fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
        frame_skip = fps  # 1 frame por segundo

        frame_count = 0
        processed_frames = 0
        face_count = 0

        while True:
            ret, im = video.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue  # pula frames intermediários

            processed_frames += 1
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            try:
                for (p, q, r, s) in faces:
                    face_count += 1
                    image = gray[q:q + s, p:p + r]
                    image = cv2.resize(image, (48, 48))
                    img = extract_features(image)
                    pred = predict_emotion(self.model, img)
                    prediction_label = labels[pred.argmax()]
                    predictions.append(prediction_label)
            except cv2.error as e:
                self.logger.error(f"OpenCV error: {e}")
                pass

        video.release()

        self.logger.info(f"Total frames in video: {frame_count}")
        self.logger.info(f"Frames actually processed: {processed_frames}")
        self.logger.info(f"Total faces detected: {face_count}")

        if not predictions:
            self.logger.warning("No faces detected or no predictions made.")

        percentages = getPercentages(predictions)
        self.logger.info(f"Percentages of emotions detected: {percentages}")
        return GetEmotionPercentagesResponse(
            Angry=percentages['Angry'],
            Disgusted=percentages['Disgusted'],
            Fearful=percentages['Fearful'],
            Happy=percentages['Happy'],
            Neutral=percentages['Neutral'],
            Sad=percentages['Sad'],
            Surprised=percentages['Surprised']
        )
