from flask import Flask
from routes.video_routes import video_routes
from utils.logger import get_logger
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv()

# Initialize logger at module level so Cloud Run deployments
# get structured logging - not just local `python app.py` runs.
logger = get_logger(__name__)

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {
        "origins": [
            "http://localhost:8080",
            "https://facial-emotion-api-990683238789.us-central1.run.app"
        ]
    }},
    supports_credentials=True
)
app.register_blueprint(video_routes)
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", False)

logger.info("Application initialised (routes registered, CORS configured)")

if __name__ == "__main__":
    logger.info("Starting development server on localhost:5000")
    # Local development server
    app.run(host="localhost", port=5000)
    # Cloud Run - uncomment for deployment
    # app.run(host="0.0.0.0", port=8080)
