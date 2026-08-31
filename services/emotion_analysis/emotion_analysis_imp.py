import os
from schemas.emotion_schema import GetEmotionPercentagesResponse
from schemas.standard_output_schema import (
    StandardizedEmotionOutput,
    AnalysisMetadata,
    EmotionEvent,
    EmotionSummary,
)
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

        last_processed_second = -1

    
        frame_count = 0
        processed_frames = 0
        face_count = 0

        while True:
            ret, im = video.read()
            if not ret:
                break

            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            current_second = int(timestamp_ms / 500 ) # 2 frame per second 

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
                    self.logger.info(f"Prediction for frame {frame_count}: {prediction_label}")
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

    def get_standardized_output(self, video_path: str, video_name: str = "") -> StandardizedEmotionOutput:
        """
        Analyze a video and return a fully standardized output including
        metadata, a chronological timeline of per-frame emotion events
        with confidence scores, and an aggregated summary.

        This method extends the existing analysis pipeline to produce
        structured, time-aware results suitable for integration into
        RUXAILAB reports and dashboards.
        """
        labels = {
            0: 'Angry', 1: 'Disgusted', 2: 'Fearful',
            3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised',
        }
        timeline: list[EmotionEvent] = []
        predictions: list[str] = []

        self.logger.info(f"[Standardized] Loading video from path: {video_path}")

        # --- Handle missing / unopenable video ---
        if not os.path.exists(video_path):
            self.logger.error(f"Video file does not exist: {video_path}")
            return StandardizedEmotionOutput(
                metadata=AnalysisMetadata(video_name=video_name or os.path.basename(video_path)),
                timeline=[],
                summary=EmotionSummary(),
            )

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            self.logger.error(f"Failed to open video file: {video_path}")
            return StandardizedEmotionOutput(
                metadata=AnalysisMetadata(video_name=video_name or os.path.basename(video_path)),
                timeline=[],
                summary=EmotionSummary(),
            )

        # Retrieve video duration from the capture object
        fps = video.get(cv2.CAP_PROP_FPS) or 1.0
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_sec = total_frame_count / fps if fps > 0 else None

        last_processed_second = -1
        frame_count = 0
        processed_frames = 0
        face_count = 0

        while True:
            ret, im = video.read()
            if not ret:
                break

            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            current_second = int(timestamp_ms / 500)  # sample ~2 frames per second

            if current_second == last_processed_second:
                continue
            last_processed_second = current_second

            frame_count += 1
            processed_frames += 1
            timestamp_sec = round(timestamp_ms / 1000.0, 3)

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            try:
                for (p, q, r, s) in faces:
                    face_count += 1
                    face_img = gray[q:q + s, p:p + r]
                    face_img = cv2.resize(face_img, (48, 48))
                    img = extract_features(face_img)
                    pred = predict_emotion(self.model, img)

                    prediction_label = labels[pred.argmax()]
                    confidence = float(pred.max())

                    predictions.append(prediction_label)
                    timeline.append(
                        EmotionEvent(
                            timestamp_sec=timestamp_sec,
                            emotion=prediction_label,
                            confidence=round(confidence, 4),
                        )
                    )
                    self.logger.info(
                        f"[Standardized] t={timestamp_sec}s  emotion={prediction_label}  "
                        f"confidence={confidence:.4f}"
                    )
            except cv2.error as e:
                self.logger.error(f"OpenCV error at t={timestamp_sec}s: {e}")

        video.release()

        self.logger.info(f"[Standardized] Total frames: {frame_count}")
        self.logger.info(f"[Standardized] Processed frames: {processed_frames}")
        self.logger.info(f"[Standardized] Faces detected: {face_count}")

        # Build aggregated summary
        percentages = getPercentages(predictions)
        summary = EmotionSummary(**percentages)

        metadata = AnalysisMetadata(
            video_name=video_name or os.path.basename(video_path),
            total_frames_processed=processed_frames,
            total_faces_detected=face_count,
            video_duration_sec=video_duration_sec,
        )

        return StandardizedEmotionOutput(
            metadata=metadata,
            timeline=timeline,
            summary=summary,
        )
