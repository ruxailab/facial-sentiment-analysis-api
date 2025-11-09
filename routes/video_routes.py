from flask import Blueprint, request, jsonify
import logging
from services.data.firebase_imp import FirebaseImp
from services.emotion_analysis.emotion_analysis_imp import EmotionsAnalysisImp
from utils.utils import delete_video
import time
from dotenv import load_dotenv
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

load_dotenv()

video_routes = Blueprint("video_routes", __name__)

# Initialize Firebase service
firebase_cred_path = "service-account.json"
storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
firebase_service = FirebaseImp(firebase_cred_path=firebase_cred_path, storage_bucket=storage_bucket)

logger = logging.getLogger(__name__)

def analyze_clip(emotion_analysis_service, video_path):
    logger.info(f"Analyzing video: {video_path}")
    try:
        result = emotion_analysis_service.get_emotion_percentages(video_path)
        logger.info(f"Emotion analysis result: {result}")
        result_dict = result if isinstance(result, dict) else result.__dict__
        return result_dict
    except Exception as e:
        logger.error(f"Failed to analyze video: {e}")
        return None

def download_and_analyze_video(video_name):
    logger.info(f"Attempting to download video: {video_name} from storage.")
    try:
        local_path = f"static/videos/{video_name}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        video_path = firebase_service.download_video_from_storage(video_name)
        logger.info(f"Video downloaded successfully to: {video_path}")
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return None

    # Ajusta FPS do vídeo para 1 frame por segundo
    try:
        clip = VideoFileClip(video_path)
        if clip.fps > 1:
            logger.warning(f"High FPS detected ({clip.fps}). Reducing to 1fps.")
            clip = clip.set_fps(1)
        trimmed_path = video_path.replace(".webm", "_trimmed.mp4")
        clip.write_videofile(trimmed_path, codec="libx264", audio=False, logger=None)
        video_path = trimmed_path
        logger.info(f"Trimmed video saved: {video_path}")
    except Exception as e:
        logger.warning(f"Failed to trim video, continuing anyway: {e}")

    # Análise de emoções
    logger.info("Initializing emotion analysis.")
    emotion_analysis_service = EmotionsAnalysisImp(model_path="models/model2/model2.h5")
    start_analysis = time.time()
    result = analyze_clip(emotion_analysis_service, video_path)
    end_analysis = time.time()
    logger.info(f"Time taken for analysis: {end_analysis - start_analysis} seconds")

    return result  # retorna o objeto de emoções diretamente

@video_routes.route("/process_video", methods=["POST"])
def process_video():
    start = time.time()
    video_name = request.json.get("video_name")
    if not video_name:
        return jsonify({"error": "Video name missing"}), 400

    try:
        result = download_and_analyze_video(video_name)
        delete_video()
    except Exception as e:
        logger.exception(f"Video processing failed: {e}")
        return jsonify({"error": "Video processing failed"}), 500

    end = time.time()
    logger.info(f"Video processed in {end-start:.2f}s")
    return jsonify({"emotions": result}), 200

@video_routes.route("/test", methods=["GET"])
def call_hello_world():
    logger.info("Attempting to call test firebase function.")
    firebase_function_url = "https://europe-west1-backend-tfg-1d0d5.cloudfunctions.net/hello_world"
    try: 
        response = request.get(firebase_function_url)
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            logger.error(f"Failed to call firebase function: {response.text}")
            return jsonify({"error": "Failed to call firebase function"}), response.status_code
    except Exception as e:
        logger.error(f"Failed to call firebase function: {e}")
        return jsonify({"error": "Failed to call firebase function"}), 500

@video_routes.route("/hello", methods=["GET"])
def hello_world():
    return jsonify({"message": "Hello World!"}), 200
