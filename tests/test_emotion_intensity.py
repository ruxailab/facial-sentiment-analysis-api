"""Unit tests for emotion intensity metrics."""

import math
import pytest
from services.metrics.emotion_intensity import (
    EmotionIntensityMetrics,
    compute_intensity_metrics,
    compute_valence,
    compute_arousal,
    compute_diversity,
    compute_dominance_ratio,
    _normalize_percentages,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def happy_dominant():
    return {
        "Happy": 60.0, "Neutral": 15.0, "Surprised": 10.0,
        "Sad": 5.0, "Angry": 5.0, "Fearful": 3.0, "Disgusted": 2.0,
    }


@pytest.fixture
def angry_dominant():
    return {
        "Angry": 50.0, "Disgusted": 15.0, "Sad": 10.0,
        "Fearful": 10.0, "Neutral": 10.0, "Happy": 3.0, "Surprised": 2.0,
    }


@pytest.fixture
def uniform_emotions():
    """Approximately uniform distribution."""
    return {
        "Happy": 14.3, "Neutral": 14.3, "Surprised": 14.3,
        "Sad": 14.3, "Angry": 14.3, "Fearful": 14.3, "Disgusted": 14.2,
    }


@pytest.fixture
def single_emotion():
    return {
        "Happy": 100.0, "Neutral": 0.0, "Surprised": 0.0,
        "Sad": 0.0, "Angry": 0.0, "Fearful": 0.0, "Disgusted": 0.0,
    }


@pytest.fixture
def all_zero():
    return {
        "Happy": 0.0, "Neutral": 0.0, "Surprised": 0.0,
        "Sad": 0.0, "Angry": 0.0, "Fearful": 0.0, "Disgusted": 0.0,
    }


# ── Normalization tests ──────────────────────────────────────────────

class TestNormalize:
    def test_sums_to_one(self, happy_dominant):
        normed = _normalize_percentages(happy_dominant)
        assert sum(normed.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_zero(self, all_zero):
        normed = _normalize_percentages(all_zero)
        assert all(v == 0.0 for v in normed.values())


# ── Valence tests ────────────────────────────────────────────────────

class TestValence:
    def test_positive_dominant_gives_positive_valence(self, happy_dominant):
        normed = _normalize_percentages(happy_dominant)
        v = compute_valence(normed)
        assert v > 0.0

    def test_negative_dominant_gives_negative_valence(self, angry_dominant):
        normed = _normalize_percentages(angry_dominant)
        v = compute_valence(normed)
        assert v < 0.0

    def test_valence_bounded(self, happy_dominant):
        normed = _normalize_percentages(happy_dominant)
        v = compute_valence(normed)
        assert -1.0 <= v <= 1.0

    def test_single_happy_gives_max_valence(self, single_emotion):
        normed = _normalize_percentages(single_emotion)
        v = compute_valence(normed)
        assert v == pytest.approx(1.0)


# ── Arousal tests ────────────────────────────────────────────────────

class TestArousal:
    def test_arousal_bounded(self, happy_dominant):
        normed = _normalize_percentages(happy_dominant)
        a = compute_arousal(normed)
        assert 0.0 <= a <= 1.0

    def test_angry_has_high_arousal(self, angry_dominant):
        normed = _normalize_percentages(angry_dominant)
        a = compute_arousal(normed)
        assert a > 0.5


# ── Diversity tests ──────────────────────────────────────────────────

class TestDiversity:
    def test_uniform_high_diversity(self, uniform_emotions):
        normed = _normalize_percentages(uniform_emotions)
        d = compute_diversity(normed)
        assert d > 0.9

    def test_single_emotion_zero_diversity(self, single_emotion):
        normed = _normalize_percentages(single_emotion)
        d = compute_diversity(normed)
        assert d == pytest.approx(0.0)

    def test_diversity_bounded(self, happy_dominant):
        normed = _normalize_percentages(happy_dominant)
        d = compute_diversity(normed)
        assert 0.0 <= d <= 1.0


# ── Dominance ratio tests ───────────────────────────────────────────

class TestDominanceRatio:
    def test_dominant_emotion_correct(self, happy_dominant):
        dom, pct, ratio = compute_dominance_ratio(happy_dominant)
        assert dom == "Happy"
        assert pct == 60.0

    def test_ratio_greater_than_one_for_clear_dominant(self, happy_dominant):
        _, _, ratio = compute_dominance_ratio(happy_dominant)
        assert ratio > 1.0

    def test_single_emotion_infinite_ratio(self, single_emotion):
        _, _, ratio = compute_dominance_ratio(single_emotion)
        assert ratio == float("inf")


# ── Full metrics computation tests ───────────────────────────────────

class TestComputeIntensityMetrics:
    def test_returns_correct_type(self, happy_dominant):
        result = compute_intensity_metrics(happy_dominant)
        assert isinstance(result, EmotionIntensityMetrics)

    def test_to_dict_keys(self, happy_dominant):
        result = compute_intensity_metrics(happy_dominant)
        d = result.to_dict()
        expected_keys = {
            "valence_score", "arousal_score", "diversity_index",
            "dominant_emotion", "dominant_percentage", "dominance_ratio",
            "interpretation",
        }
        assert set(d.keys()) == expected_keys

    def test_interpretation_is_string(self, happy_dominant):
        result = compute_intensity_metrics(happy_dominant)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 20

    def test_all_zero_handled(self, all_zero):
        result = compute_intensity_metrics(all_zero)
        assert result.valence_score == 0.0
        assert result.arousal_score == 0.0
        assert result.diversity_index == 0.0

    def test_positive_valence_for_happy(self, happy_dominant):
        result = compute_intensity_metrics(happy_dominant)
        assert result.valence_score > 0.3

    def test_negative_valence_for_angry(self, angry_dominant):
        result = compute_intensity_metrics(angry_dominant)
        assert result.valence_score < -0.3
