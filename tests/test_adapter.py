"""
Unit tests for the facial emotion normalization adapter.
"""

import json
import pytest

from schemas.emotion_schema import GetEmotionPercentagesResponse
from normalization.schema import StandardizedOutput, ResultEntry, Segment
from normalization.adapter import (
    normalize_emotions,
    EMOTION_LABEL_MAP,
    _now_iso,
)


# ------------------------------------------------------------------ #
# Schema model tests
# ------------------------------------------------------------------ #

class TestSchemaModels:
    """Verify the Pydantic schema models behave correctly."""

    def test_result_entry_minimal(self):
        entry = ResultEntry(label="happy", score=0.75)
        assert entry.label == "happy"
        assert entry.score == 0.75
        assert entry.intensity is None
        assert entry.segment is None

    def test_result_entry_with_segment(self):
        seg = Segment(start=0.0, end=5.0, text="sample")
        entry = ResultEntry(label="neutral", score=0.5, segment=seg)
        assert entry.segment.start == 0.0
        assert entry.segment.text == "sample"

    def test_result_entry_with_intensity(self):
        entry = ResultEntry(label="angry", score=0.9, intensity="high")
        assert entry.intensity == "high"

    def test_score_upper_bound(self):
        with pytest.raises(Exception):
            ResultEntry(label="happy", score=1.5)

    def test_score_lower_bound(self):
        with pytest.raises(Exception):
            ResultEntry(label="happy", score=-0.1)

    def test_standardized_output_serialization(self):
        output = StandardizedOutput(
            analysis_type="emotion",
            modality="facial",
            source_model="test-model",
            timestamp="2026-03-09T12:00:00Z",
            input_summary="video.mp4",
            results=[ResultEntry(label="happy", score=0.6)],
        )
        data = output.model_dump()
        assert data["schema_version"] == "1.0"
        assert data["analysis_type"] == "emotion"
        assert data["modality"] == "facial"
        assert len(data["results"]) == 1

    def test_standardized_output_to_json(self):
        output = StandardizedOutput(
            analysis_type="emotion",
            modality="facial",
            source_model="test-model",
            timestamp="2026-03-09T12:00:00Z",
            input_summary="video.mp4",
            results=[
                ResultEntry(label="happy", score=0.5),
                ResultEntry(label="neutral", score=0.3),
            ],
        )
        json_str = output.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["modality"] == "facial"
        assert len(parsed["results"]) == 2


# ------------------------------------------------------------------ #
# Adapter tests — dict input (percentage range 0-100)
# ------------------------------------------------------------------ #

class TestNormalizeEmotionsFromDict:
    """Tests for normalize_emotions with raw dict input."""

    def setup_method(self):
        self.emotions = {
            "Angry": 4.0,
            "Disgusted": 1.5,
            "Fearful": 2.0,
            "Happy": 45.0,
            "Neutral": 30.0,
            "Sad": 6.0,
            "Surprised": 12.0,
        }

    def test_basic_conversion(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.analysis_type == "emotion"
        assert result.modality == "facial"
        assert result.source_model == "facial-emotion-model2"
        assert result.input_summary == "recording.mp4"
        assert len(result.results) == 7

    def test_labels_lowercased(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        for entry in result.results:
            assert entry.label == entry.label.lower()

    def test_scores_normalized_to_0_1(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        for entry in result.results:
            assert 0.0 <= entry.score <= 1.0

    def test_sorted_by_score_descending(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_dominant_emotion_first(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.results[0].label == "happy"
        assert result.results[0].score == 0.45

    def test_task_id_included(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            task_id="session-42",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.task_id == "session-42"

    def test_task_id_default_none(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.task_id is None

    def test_custom_source_model(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            source_model="emotiondetector-model1",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.source_model == "emotiondetector-model1"

    def test_auto_timestamp(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
        )
        assert result.timestamp is not None
        assert "T" in result.timestamp

    def test_explicit_timestamp(self):
        ts = "2026-01-15T08:30:00Z"
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp=ts,
        )
        assert result.timestamp == ts

    def test_schema_version(self):
        result = normalize_emotions(
            emotion_data=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.schema_version == "1.0"


# ------------------------------------------------------------------ #
# Adapter tests — dict input (scores already in 0-1 range)
# ------------------------------------------------------------------ #

class TestNormalizeEmotionsAlreadyNormalized:
    """Tests when scores are already in 0-1 range."""

    def test_scores_kept_as_is(self):
        emotions_01 = {
            "Happy": 0.45,
            "Neutral": 0.30,
            "Surprised": 0.12,
            "Sad": 0.06,
            "Angry": 0.04,
            "Fearful": 0.02,
            "Disgusted": 0.01,
        }
        result = normalize_emotions(
            emotion_data=emotions_01,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        labels_scores = {r.label: r.score for r in result.results}
        assert labels_scores["happy"] == 0.45
        assert labels_scores["neutral"] == 0.30
        assert labels_scores["disgusted"] == 0.01

    def test_boundary_score_of_1(self):
        emotions = {
            "Happy": 1.0,
            "Neutral": 0.0,
            "Surprised": 0.0,
            "Sad": 0.0,
            "Angry": 0.0,
            "Fearful": 0.0,
            "Disgusted": 0.0,
        }
        result = normalize_emotions(
            emotion_data=emotions,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        labels_scores = {r.label: r.score for r in result.results}
        assert labels_scores["happy"] == 1.0
        assert labels_scores["neutral"] == 0.0


# ------------------------------------------------------------------ #
# Adapter tests — GetEmotionPercentagesResponse input
# ------------------------------------------------------------------ #

class TestNormalizeEmotionsFromPydantic:
    """Tests using the actual Pydantic response model as input."""

    def test_from_pydantic_model(self):
        response = GetEmotionPercentagesResponse(
            Angry=10.0,
            Disgusted=5.0,
            Fearful=3.0,
            Happy=50.0,
            Neutral=20.0,
            Sad=7.0,
            Surprised=5.0,
        )
        result = normalize_emotions(
            emotion_data=response,
            video_path="task_video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.analysis_type == "emotion"
        assert len(result.results) == 7
        assert result.results[0].label == "happy"
        assert result.results[0].score == 0.5

    def test_pydantic_model_all_zero(self):
        response = GetEmotionPercentagesResponse(
            Angry=0.0,
            Disgusted=0.0,
            Fearful=0.0,
            Happy=0.0,
            Neutral=0.0,
            Sad=0.0,
            Surprised=0.0,
        )
        result = normalize_emotions(
            emotion_data=response,
            video_path="empty.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        for entry in result.results:
            assert entry.score == 0.0

    def test_pydantic_model_preserves_all_emotions(self):
        response = GetEmotionPercentagesResponse(
            Angry=14.29,
            Disgusted=14.29,
            Fearful=14.28,
            Happy=14.29,
            Neutral=14.29,
            Sad=14.28,
            Surprised=14.28,
        )
        result = normalize_emotions(
            emotion_data=response,
            video_path="balanced.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        labels = {r.label for r in result.results}
        expected = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}
        assert labels == expected


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #

class TestEdgeCases:
    """Edge cases and unusual inputs."""

    def test_unknown_emotion_label(self):
        """An unexpected label should be lowercased gracefully."""
        emotions = {
            "Angry": 10.0,
            "Disgusted": 5.0,
            "Fearful": 3.0,
            "Happy": 40.0,
            "Neutral": 20.0,
            "Sad": 7.0,
            "Surprised": 5.0,
            "Contempt": 10.0,
        }
        result = normalize_emotions(
            emotion_data=emotions,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        labels = {r.label for r in result.results}
        assert "contempt" in labels

    def test_empty_dict(self):
        result = normalize_emotions(
            emotion_data={},
            video_path="nothing.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert len(result.results) == 0

    def test_single_dominant_emotion(self):
        emotions = {
            "Angry": 0.0,
            "Disgusted": 0.0,
            "Fearful": 0.0,
            "Happy": 100.0,
            "Neutral": 0.0,
            "Sad": 0.0,
            "Surprised": 0.0,
        }
        result = normalize_emotions(
            emotion_data=emotions,
            video_path="happy_only.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.results[0].label == "happy"
        assert result.results[0].score == 1.0
        for entry in result.results[1:]:
            assert entry.score == 0.0

    def test_output_is_valid_json(self):
        emotions = {
            "Angry": 15.5,
            "Disgusted": 2.1,
            "Fearful": 8.3,
            "Happy": 45.2,
            "Neutral": 20.0,
            "Sad": 5.1,
            "Surprised": 3.8,
        }
        result = normalize_emotions(
            emotion_data=emotions,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0"
        assert parsed["analysis_type"] == "emotion"
        assert parsed["modality"] == "facial"
        assert len(parsed["results"]) == 7

    def test_no_segments_in_facial_output(self):
        """Facial emotion results should not include segment data."""
        emotions = {"Happy": 50.0, "Neutral": 50.0}
        result = normalize_emotions(
            emotion_data=emotions,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        for entry in result.results:
            assert entry.segment is None

    def test_no_intensity_by_default(self):
        """Facial emotion results should not include intensity by default."""
        emotions = {"Happy": 50.0, "Neutral": 50.0}
        result = normalize_emotions(
            emotion_data=emotions,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        for entry in result.results:
            assert entry.intensity is None


# ------------------------------------------------------------------ #
# Label map completeness
# ------------------------------------------------------------------ #

class TestLabelMap:
    """Verify the label mapping covers all expected emotions."""

    def test_all_seven_emotions_mapped(self):
        expected = {
            "Angry", "Disgusted", "Fearful",
            "Happy", "Neutral", "Sad", "Surprised",
        }
        assert set(EMOTION_LABEL_MAP.keys()) == expected

    def test_all_values_are_lowercase(self):
        for value in EMOTION_LABEL_MAP.values():
            assert value == value.lower()

    def test_map_matches_schema_emotions(self):
        """The mapped values should match the existing GetEmotionPercentagesResponse fields."""
        schema_fields = set(GetEmotionPercentagesResponse.model_fields.keys())
        map_keys = set(EMOTION_LABEL_MAP.keys())
        assert schema_fields == map_keys


# ------------------------------------------------------------------ #
# Timestamp helper
# ------------------------------------------------------------------ #

class TestTimestampHelper:
    """Verify the ISO timestamp utility."""

    def test_now_iso_format(self):
        ts = _now_iso()
        assert "T" in ts
        assert ts.endswith("Z")
        assert len(ts) == 20
