import os
from collections import defaultdict
from schemas.emotion_schema import GetEmotionPercentagesResponse, TimelineEntry, EmotionAnalysisResponse
from services.emotion_analysis.emotion_analysis_service import EmotionsAnalysisService
import logging
import coloredlogs
from utils.utils import load_model, load_face_cascade, extract_features, predict_emotion, getPercentages
import cv2
import numpy as np

EMOTION_LABELS = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}


def _format_time(seconds):
    return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"


def _get_dominant(preds):
    """Return (dominant_label, avg_confidence_%) from list of (label, confidence) tuples."""
    if not preds:
        return None, 0.0
    counts, confs = defaultdict(int), defaultdict(list)
    for label, conf in preds:
        counts[label] += 1
        confs[label].append(conf)
    dominant = max(counts, key=counts.get)
    return dominant, round(np.mean(confs[dominant]) * 100, 2)


def _build_timeline_fixed(timed_preds, interval_s, duration):
    """Group predictions into fixed time windows, return dominant emotion per window."""
    if not timed_preds or duration <= 0:
        return []
    num_windows = max(1, int(duration / interval_s) + (1 if duration % interval_s else 0))
    windows = defaultdict(list)
    for ts, label, conf in timed_preds:
        windows[min(int(ts / interval_s), num_windows - 1)].append((label, conf))

    timeline = []
    for i in range(num_windows):
        if i not in windows:
            continue
        dominant, avg_conf = _get_dominant(windows[i])
        timeline.append(TimelineEntry(
            id=len(timeline) + 1,
            starting_time=_format_time(i * interval_s),
            ending_time=_format_time(min((i + 1) * interval_s, duration)),
            emotion=dominant.upper(), value=avg_conf,
        ))
    return timeline


def _build_timeline_dynamic(timed_preds, duration):
    """Segment timeline whenever the dominant emotion changes."""
    if not timed_preds or duration <= 0:
        return []
    timeline, seg_preds = [], []
    cur_label, seg_start = timed_preds[0][1], timed_preds[0][0]

    for ts, label, conf in timed_preds:
        if label != cur_label:
            _, avg_conf = _get_dominant(seg_preds)
            timeline.append(TimelineEntry(
                id=len(timeline) + 1, starting_time=_format_time(seg_start),
                ending_time=_format_time(ts), emotion=cur_label.upper(), value=avg_conf,
            ))
            cur_label, seg_start, seg_preds = label, ts, []
        seg_preds.append((label, conf))

    if seg_preds:
        _, avg_conf = _get_dominant(seg_preds)
        timeline.append(TimelineEntry(
            id=len(timeline) + 1, starting_time=_format_time(seg_start),
            ending_time=_format_time(duration), emotion=cur_label.upper(), value=avg_conf,
        ))
    return timeline


class EmotionsAnalysisImp(EmotionsAnalysisService):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.face_cascade = load_face_cascade()
        coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def get_emotion_percentages(self, video_path: str, interval_s: int = 10) -> EmotionAnalysisResponse:
        predictions, timed_predictions = [], []
        self.logger.info(f"Loading video from path: {video_path}")

        empty = EmotionAnalysisResponse(
            emotions=GetEmotionPercentagesResponse(
                Angry=0, Disgusted=0, Fearful=0, Happy=0, Neutral=0, Sad=0, Surprised=0),
            timeline=[],
        )

        if not os.path.exists(video_path):
            self.logger.error(f"Video file does not exist: {video_path}")
            directory = os.path.dirname(video_path)
            if os.path.exists(directory):
                self.logger.info(f"Contents of {directory}: {os.listdir(directory)}")
            return empty

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            self.logger.error(f"Failed to open video file: {video_path}")
            return empty

        fps = video.get(cv2.CAP_PROP_FPS)
        video_duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0
        last_processed_second = -1
        frame_count, face_count = 0, 0

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
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            try:
                for (p, q, r, s) in faces:
                    face_count += 1
                    image = cv2.resize(gray[q:q+s, p:p+r], (48, 48))
                    pred = predict_emotion(self.model, extract_features(image))
                    pred_idx = pred.argmax()
                    label = EMOTION_LABELS[pred_idx]
                    conf = float(pred[0][pred_idx])
                    predictions.append(label)
                    timed_predictions.append((timestamp_ms / 1000.0, label, conf))
            except cv2.error as e:
                self.logger.error(f"OpenCV error: {e}")

        video.release()
        self.logger.info(f"Processed {frame_count} frames, {face_count} faces detected")

        if not predictions:
            self.logger.warning("No faces detected or no predictions made.")

        percentages = getPercentages(predictions)
        emotions = GetEmotionPercentagesResponse(**percentages)

        if interval_s == 0:
            timeline = _build_timeline_dynamic(timed_predictions, video_duration)
        else:
            timeline = _build_timeline_fixed(timed_predictions, interval_s, video_duration)
        self.logger.info(f"Timeline: {len(timeline)} entries")

        return EmotionAnalysisResponse(emotions=emotions, timeline=timeline)
