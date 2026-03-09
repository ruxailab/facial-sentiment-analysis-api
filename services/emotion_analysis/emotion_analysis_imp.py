import os
import cv2
import logging
import coloredlogs
import numpy as np
from insightface.app import FaceAnalysis
from schemas.emotion_schema import GetEmotionPercentagesResponse
from services.emotion_analysis.emotion_analysis_service import EmotionsAnalysisService
from utils.utils import load_model, load_face_cascade, extract_features, predict_emotion, getPercentages

coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class EmotionsAnalysisImp(EmotionsAnalysisService):
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        self.model = load_model(model_path)
        try:
            self.scrfd_model = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            self.scrfd_model.prepare(ctx_id=0, det_size=(640,640))
            self.logger.info("SCRFD Face detector loaded successfully (buffalo_s)")
        except Exception as e:
            self.logger.error(f"Failed to load SCRFD face detector: {e}")

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

        last_processed_second = -1

        frame_count = 0
        processed_frames = 0
        face_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            current_second = int(timestamp_ms / 500)  # 2 frames per second
            if current_second == last_processed_second:
                continue
            last_processed_second = current_second

            frame_count += 1
            processed_frames += 1

            scale_factor = 1.5
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            faces = self.scrfd_model.get(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                det_score = getattr(face, 'det_score', 0)
                if det_score < 0.5:
                    continue

                margin_w = int((x2 - x1) * 0.4)
                margin_h = int((y2 - y1) * 0.4)
                x1_new = max(0, x1 - margin_w)
                y1_new = max(0, y1 - margin_h)
                x2_new = min(frame.shape[1], x2 + margin_w)
                y2_new = min(frame.shape[0], y2 + margin_h)

                face_count += 1
                face_crop = frame[y1_new:y2_new, x1_new:x2_new]

                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(gray_face, (48, 48))
                img = extract_features(face_resized)

                try:
                    pred = predict_emotion(self.model, img)
                    prediction_label = labels[pred.argmax()]
                    predictions.append(prediction_label)
                    self.logger.info(f"Frame {frame_count}: Detected emotion: {prediction_label}")
                except Exception as e:
                    self.logger.error(f"Error predicting emotion for frame {frame_count}: {e}")
                    continue

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


