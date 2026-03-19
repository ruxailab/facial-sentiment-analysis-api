"""Tests for timeline building logic."""
import sys
from unittest.mock import MagicMock
from types import ModuleType
import pytest

# Mock heavy deps
for mod in ['cv2', 'tensorflow', 'tensorflow.keras', 'tensorflow.keras.models',
            'coloredlogs', 'moviepy', 'moviepy.video', 'moviepy.video.io',
            'moviepy.video.io.ffmpeg_tools', 'moviepy.editor']:
    sys.modules[mod] = MagicMock()
mock_utils = ModuleType('utils.utils')
mock_utils.load_model = mock_utils.load_face_cascade = MagicMock()
mock_utils.extract_features = mock_utils.predict_emotion = MagicMock()
mock_utils.getPercentages = MagicMock(return_value={
    'Angry': 0, 'Disgusted': 0, 'Fearful': 0,
    'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprised': 0})
sys.modules['utils'] = ModuleType('utils')
sys.modules['utils.utils'] = mock_utils

from services.emotion_analysis.emotion_analysis_imp import (
    _build_timeline_fixed, _build_timeline_dynamic, _format_time, _get_dominant,
)


def test_format_time():
    assert _format_time(0) == "00:00"
    assert _format_time(75) == "01:15"


def test_get_dominant():
    label, conf = _get_dominant([("Happy", 0.9), ("Happy", 0.8), ("Sad", 0.7)])
    assert label == "Happy"
    assert conf == 85.0
    assert _get_dominant([]) == (None, 0.0)


def test_fixed_timeline_multiple_windows():
    preds = [(2.0, "Happy", 0.9), (12.0, "Sad", 0.8), (25.0, "Angry", 0.7)]
    timeline = _build_timeline_fixed(preds, 10, 30)
    assert len(timeline) == 3
    assert [t.emotion for t in timeline] == ["HAPPY", "SAD", "ANGRY"]
    assert timeline[0].starting_time == "00:00"


def test_fixed_timeline_skips_empty_windows():
    timeline = _build_timeline_fixed([(2.0, "Happy", 0.9), (25.0, "Sad", 0.8)], 10, 30)
    assert len(timeline) == 2


def test_fixed_timeline_confidence_value():
    timeline = _build_timeline_fixed([(1.0, "Happy", 0.92), (3.0, "Happy", 0.88)], 10, 10)
    assert timeline[0].value == 90.0


def test_dynamic_timeline_segments_on_change():
    preds = [(0.0, "Happy", 0.9), (3.0, "Happy", 0.85),
             (6.0, "Sad", 0.8), (12.0, "Angry", 0.7)]
    timeline = _build_timeline_dynamic(preds, 15)
    assert len(timeline) == 3
    assert [t.emotion for t in timeline] == ["HAPPY", "SAD", "ANGRY"]


def test_empty_inputs():
    assert _build_timeline_fixed([], 10, 30) == []
    assert _build_timeline_dynamic([], 30) == []
