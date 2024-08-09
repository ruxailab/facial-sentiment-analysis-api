from flask import Blueprint, request, jsonify
import logging
from services.data.firebase_imp import FirebaseImp
from services.emotion_analysis.emotion_analysis_imp import EmotionsAnalysisImp
from utils.utils import delete_video, split_video_into_clips
import time

video_routes = Blueprint("video_routes", __name__)

# Initialize Firebase service
firebase_cred_path = "service-account.json"
storage_bucket = "retlab-dev.appspot.com"
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
    # Download the video from Firebase Storage
    logger.info(f"Attempting to download video: {video_name} from storage.")
    try:
        video_path = firebase_service.download_video_from_storage(video_name)
        logger.info(f"Video downloaded successfully to: {video_path}")
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return
    
    # Split video into clips
    logger.info(f"Splitting video into clips: {video_path}")
    video_paths = split_video_into_clips(video_path)
    logger.info(f"Split video clips: {video_paths}")

    logger.info("Initializing emotion analysis.")
    emotion_analysis_service = EmotionsAnalysisImp(model_path="models/model2/model2.h5")
    results = []

    # Process the video clips sequentially
    start_analysis = time.time()
    for video_path in video_paths:
        result = analyze_clip(emotion_analysis_service, video_path)
        if result:
            results.append(result)
    end_analysis = time.time()
    logger.info(f"Time taken for analysis: {end_analysis - start_analysis} seconds")

    # Upload the analysis results to Firestore
    doc_id = 0
    try:
        doc_id = firebase_service.upload_to_firestore(results)
        logger.info(f"Analysis results uploaded successfully. Document ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to upload analysis results to Firestore: {e}")
    return doc_id

@video_routes.route("/process_video", methods=["POST"])
def process_video():
    # Start timer
    start = time.time()
    
    # Assuming the request contains the name of the video to process
    video_name = request.json.get("video_name")
    if not video_name:
        return jsonify({"error": "Video name is missing in the request"}), 400
    
    # Process the video
    doc_id= download_and_analyze_video(video_name)
    
    logger.info("Deleting video from local storage.")
    delete_video()
    
    # End timer
    end = time.time()
    logger.info(f"Time taken to process video: {end - start} seconds")
    
    #return a message indicating the video was processed and the document ID
    return jsonify({"message": "Video processed successfully", "doc_id": doc_id}), 200
    
@video_routes.route("/test", methods=["GET"])
def call_hello_world():
    logger.info("Attempting to call test firebase function.")
    # Using emulators: http://127.0.0.1:5001/backend-tfg-1d0d5/europe-west1/hello_world
    # Without emulators: https://europe-west1-backend-tfg-1d0d5.cloudfunctions.net/hello_world
    firebase_function_url = "https://europe-west1-backend-tfg-1d0d5.cloudfunctions.net/hello_world"
    try: 
        response = request.get(firebase_function_url)  # Allows calling the firebase function hosted elsewhere, available methods: get, post, put, delete
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            logger.error(f"Failed to call firebase function: {response.text}")
            return jsonify({"error": "Failed to call firebase function"}), response.status_code
    except Exception as e:
        logger.error(f"Failed to call firebase function: {e}")
        return jsonify({"error": "Failed to call firebase function"}), 500

# Endpoint that returns hello for testing 
@video_routes.route("/hello", methods=["GET"])
def hello_world():
    return jsonify({"message": "Hello World!"}), 200