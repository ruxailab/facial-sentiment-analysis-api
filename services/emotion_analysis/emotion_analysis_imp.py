import os
from schemas.emotion_schema import GetEmotionPercentagesResponse
from services.emotion_analysis.emotion_analysis_service import EmotionsAnalysisService
from utils.logger import get_logger
from utils.utils import load_model, load_face_cascade, extract_features, predict_emotion, getPercentages
import cv2

logger = get_logger(__name__)


class EmotionsAnalysisImp(EmotionsAnalysisService):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.face_cascade = load_face_cascade()

    def get_emotion_percentages(self, video_path: str) -> GetEmotionPercentagesResponse:
        predictions = []
        labels = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}
        logger.info("Loading video from path: %s", video_path)

        if not os.path.exists(video_path):
            logger.error("Video file does not exist: %s", video_path)
            directory = os.path.dirname(video_path)
            if os.path.exists(directory):
                logger.info("Contents of directory %s:", directory)
                for item in os.listdir(directory):
                    logger.info("  - %s", item)
            else:
                logger.error("Directory does not exist: %s", directory)
            return GetEmotionPercentagesResponse(
                Angry=0, Disgusted=0, Fearful=0, Happy=0, Neutral=0, Sad=0, Surprised=0
            )

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error("Failed to open video file: %s", video_path)
            return GetEmotionPercentagesResponse(
                Angry=0, Disgusted=0, Fearful=0, Happy=0, Neutral=0, Sad=0, Surprised=0
            )

        last_processed_second = -1
        frame_count = 0
        processed_frames = 0
        face_count = 0

        while True:
            ret, im = video.read()
            if not ret:
                break

            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            current_second = int(timestamp_ms / 500)  # 2 frames per second

            if current_second == last_processed_second:
                continue
            last_processed_second = current_second

            frame_count += 1
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
                    logger.info("Prediction for frame %d: %s", frame_count, prediction_label)
                    predictions.append(prediction_label)
            except cv2.error as e:
                logger.error("OpenCV error on frame %d: %s", frame_count, e)

        video.release()

        logger.info("Total frames in video: %d", frame_count)
        logger.info("Frames actually processed: %d", processed_frames)
        logger.info("Total faces detected: %d", face_count)

        if not predictions:
            logger.warning("No faces detected or no predictions made.")

        percentages = getPercentages(predictions)
        logger.info("Emotion percentages: %s", percentages)
        return GetEmotionPercentagesResponse(
            Angry=percentages['Angry'],
            Disgusted=percentages['Disgusted'],
            Fearful=percentages['Fearful'],
            Happy=percentages['Happy'],
            Neutral=percentages['Neutral'],
            Sad=percentages['Sad'],
            Surprised=percentages['Surprised']
        )
