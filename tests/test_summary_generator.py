"""Unit tests for the human-readable summary generator."""

import pytest
from services.report.summary_generator import (
    EmotionBreakdown,
    generate_summary,
    _sentiment_label,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def happy_dominant():
    return {
        "Happy": 55.0, "Neutral": 20.0, "Surprised": 10.0,
        "Sad": 5.0, "Angry": 5.0, "Fearful": 3.0, "Disgusted": 2.0,
    }


@pytest.fixture
def angry_dominant():
    return {
        "Angry": 45.0, "Disgusted": 15.0, "Sad": 10.0,
        "Fearful": 5.0, "Neutral": 15.0, "Happy": 5.0, "Surprised": 5.0,
    }


@pytest.fixture
def neutral_dominant():
    return {
        "Neutral": 70.0, "Happy": 10.0, "Sad": 5.0,
        "Angry": 5.0, "Fearful": 5.0, "Disgusted": 3.0, "Surprised": 2.0,
    }


@pytest.fixture
def mixed_emotions():
    return {
        "Happy": 25.0, "Sad": 25.0, "Neutral": 20.0,
        "Angry": 10.0, "Fearful": 10.0, "Disgusted": 5.0, "Surprised": 5.0,
    }


@pytest.fixture
def all_zero():
    return {
        "Happy": 0.0, "Sad": 0.0, "Neutral": 0.0,
        "Angry": 0.0, "Fearful": 0.0, "Disgusted": 0.0, "Surprised": 0.0,
    }


# ── EmotionBreakdown tests ───────────────────────────────────────────

class TestEmotionBreakdown:
    def test_dominant_emotion(self, happy_dominant):
        bd = EmotionBreakdown(percentages=happy_dominant)
        assert bd.dominant_emotion == "Happy"
        assert bd.dominant_percentage == 55.0

    def test_positive_total(self, happy_dominant):
        bd = EmotionBreakdown(percentages=happy_dominant)
        assert bd.positive_total == pytest.approx(65.0)  # Happy 55 + Surprised 10

    def test_negative_total(self, angry_dominant):
        bd = EmotionBreakdown(percentages=angry_dominant)
        assert bd.negative_total == pytest.approx(75.0)  # Angry 45 + Disgusted 15 + Sad 10 + Fearful 5

    def test_top_n(self, happy_dominant):
        bd = EmotionBreakdown(percentages=happy_dominant)
        top3 = bd.top_n(3)
        assert len(top3) == 3
        assert top3[0][0] == "Happy"


# ── Sentiment label tests ────────────────────────────────────────────

class TestSentimentLabel:
    def test_positive(self, happy_dominant):
        bd = EmotionBreakdown(percentages=happy_dominant)
        label = _sentiment_label(bd)
        assert "Positive" in label  # "Positive" or "Mostly Positive"

    def test_negative(self, angry_dominant):
        bd = EmotionBreakdown(percentages=angry_dominant)
        label = _sentiment_label(bd)
        assert "Negative" in label

    def test_neutral(self, neutral_dominant):
        bd = EmotionBreakdown(percentages=neutral_dominant)
        assert _sentiment_label(bd) == "Neutral"

    def test_mixed(self, mixed_emotions):
        bd = EmotionBreakdown(percentages=mixed_emotions)
        label = _sentiment_label(bd)
        assert label in ("Mixed", "Mostly Negative", "Mostly Positive")


# ── Full summary generation tests ────────────────────────────────────

class TestGenerateSummary:
    def test_returns_all_keys(self, happy_dominant):
        result = generate_summary(happy_dominant)
        assert "overall_sentiment" in result
        assert "top_emotions" in result
        assert "ux_insight" in result
        assert "full_summary" in result

    def test_video_name_in_summary(self, happy_dominant):
        result = generate_summary(happy_dominant, video_name="session_42.webm")
        assert "session_42.webm" in result["full_summary"]

    def test_no_video_name(self, happy_dominant):
        result = generate_summary(happy_dominant)
        assert "Analysis of" not in result["full_summary"]

    def test_all_zero_emotions(self, all_zero):
        result = generate_summary(all_zero)
        assert "No emotions were detected" in result["top_emotions"]

    def test_positive_insight(self, happy_dominant):
        result = generate_summary(happy_dominant)
        assert "positive" in result["ux_insight"].lower()

    def test_negative_insight(self, angry_dominant):
        result = generate_summary(angry_dominant)
        assert any(word in result["ux_insight"].lower() for word in ["frustration", "negative", "anger"])

    def test_summary_is_string(self, happy_dominant):
        result = generate_summary(happy_dominant)
        assert isinstance(result["full_summary"], str)
        assert len(result["full_summary"]) > 50
