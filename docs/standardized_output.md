# Standardized Emotion Output Adapter

## Purpose

This module provides a lightweight adapter that converts the facial emotion analysis results produced by this API into the **RUXAILAB Standardized Output Schema (v1.0)**.

The standardized schema was proposed in the sibling repository [`sentiment-analysis-api`](https://github.com/ruxailab/sentiment-analysis-api) to unify outputs across different analysis services (text sentiment, audio transcript sentiment, and facial emotion detection). By producing the same structure, any downstream consumer (frontend dashboards, Firestore storage, cross-modality comparison tools) can process results from this API without format-specific logic.

## How It Works

The adapter lives in the `normalization/` package:

- **`normalization/schema.py`** — Pydantic models defining the standardized schema (`StandardizedOutput`, `ResultEntry`, `Segment`).
- **`normalization/adapter.py`** — The `normalize_emotions()` function that performs the conversion.

### Conversion steps

1. Accepts a `GetEmotionPercentagesResponse` object (or equivalent dict) as returned by `EmotionsAnalysisImp.get_emotion_percentages()`.
2. Maps emotion labels from title-case (`Happy`, `Angry`, …) to lowercase (`happy`, `angry`, …).
3. Converts percentage scores (0–100) to normalized scores (0.0–1.0). Scores already in the 0–1 range are kept as-is.
4. Sorts results by score in descending order so the dominant emotion appears first.
5. Returns a validated `StandardizedOutput` instance.

### Usage example

```python
from schemas.emotion_schema import GetEmotionPercentagesResponse
from normalization.adapter import normalize_emotions

# After running emotion analysis on a video...
percentages = GetEmotionPercentagesResponse(
    Angry=4.0, Disgusted=1.0, Fearful=2.0,
    Happy=45.0, Neutral=30.0, Sad=6.0, Surprised=12.0,
)

standardized = normalize_emotions(
    emotion_data=percentages,
    video_path="recording_task3.mp4",
    task_id="usability-session-42",
)

# Serialize to JSON for storage or API response
print(standardized.model_dump_json(indent=2))
```

### Output

```json
{
  "schema_version": "1.0",
  "analysis_type": "emotion",
  "modality": "facial",
  "source_model": "facial-emotion-model2",
  "timestamp": "2026-03-09T14:30:00Z",
  "task_id": "usability-session-42",
  "input_summary": "recording_task3.mp4",
  "results": [
    { "label": "happy",     "score": 0.45, "intensity": null, "segment": null },
    { "label": "neutral",   "score": 0.30, "intensity": null, "segment": null },
    { "label": "surprised", "score": 0.12, "intensity": null, "segment": null },
    { "label": "sad",       "score": 0.06, "intensity": null, "segment": null },
    { "label": "angry",     "score": 0.04, "intensity": null, "segment": null },
    { "label": "fearful",   "score": 0.02, "intensity": null, "segment": null },
    { "label": "disgusted", "score": 0.01, "intensity": null, "segment": null }
  ]
}
```

## Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `string` | Always `"1.0"` for this version. |
| `analysis_type` | `string` | `"emotion"` for facial analysis. |
| `modality` | `string` | `"facial"` for this API. |
| `source_model` | `string` | Identifier of the model used (default: `"facial-emotion-model2"`). |
| `timestamp` | `string` | ISO 8601 datetime of when the analysis was performed. |
| `task_id` | `string \| null` | Optional usability test task or session identifier. |
| `input_summary` | `string` | Path or identifier of the analyzed video. |
| `results` | `array` | Emotion entries sorted by score descending. |

Each result entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `label` | `string` | Lowercase emotion label (`happy`, `sad`, `angry`, etc.). |
| `score` | `float` | Normalized score between 0.0 and 1.0. |
| `intensity` | `string \| null` | Reserved for future use. |
| `segment` | `object \| null` | Reserved for future per-segment analysis. |

## Integration with RUXAILAB

This adapter enables:

- **Unified frontend components**: The RUXAILAB frontend can render facial emotion results using the same components that display text and audio sentiment.
- **Consistent Firestore storage**: Results from all modalities share the same document structure.
- **Cross-modality comparison**: Standardized scores make it possible to compare or aggregate results across text, audio, and facial analysis for the same usability session.
- **Task-level linking**: The optional `task_id` field connects results to specific usability test tasks, enabling temporal correlation across modalities.

## Running Tests

```bash
pytest tests/test_adapter.py -v
```
