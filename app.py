from flask import Flask
from routes.video_routes import video_routes
import logging
import coloredlogs
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv()

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

if __name__ == "__main__":
    coloredlogs.install(
        level="INFO",
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting the application")

    # 🔁 Local
    app.run(host="localhost", port=5000)

    # ☁️ Cloud Run (comentado localmente)
    # app.run(host="0.0.0.0", port=8080)
