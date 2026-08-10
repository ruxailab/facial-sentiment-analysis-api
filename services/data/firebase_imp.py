import os
import firebase_admin
import logging
import coloredlogs

from firebase_admin import firestore, storage, credentials
from services.data.firebase_service import FirebaseService


class FirebaseImp(FirebaseService):

    logger = logging.getLogger(__name__)

    def __init__(self, storage_bucket: str):
        self.storage_bucket = storage_bucket
        self._initialize_app()

        coloredlogs.install(
            level="INFO",
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.logger = logging.getLogger(__name__)
        self.db = firestore.client()
        self.storage_client = storage.bucket()

    def _initialize_app(self):
        use_storage_emulator = os.getenv('USE_STORAGE_EMULATOR', '').lower() == 'true'
        use_firestore_emulator = os.getenv('USE_FIRESTORE_EMULATOR', '').lower() == 'true'

        emulator_host = os.getenv('FIREBASE_EMULATOR_HOST', 'localhost')

        if use_storage_emulator:
            storage_port = os.getenv('STORAGE_EMULATOR_PORT', '9199')
            os.environ['STORAGE_EMULATOR_HOST'] = f"http://{emulator_host}:{storage_port}"

        if use_firestore_emulator:
            firestore_port = os.getenv('FIRESTORE_EMULATOR_PORT', '8080')
            os.environ['FIRESTORE_EMULATOR_HOST'] = f"{emulator_host}:{firestore_port}"

        options = {
            "storageBucket": self.storage_bucket
        }

        if use_storage_emulator or use_firestore_emulator:
            options["projectId"] = os.getenv('FIREBASE_PROJECT_ID')

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                options=options
            )

    def download_video_from_storage(self, video_name: str):
        self.logger.info(f"Attempting to download video: {video_name} from storage.")

        blob = self.storage_client.blob(video_name)
        video_path = f"static/videos/{video_name}"
        blob.download_to_filename(video_path)

        self.logger.info(f"Video downloaded successfully to: {video_path}")
        return video_path

    def upload_to_firestore(self, data):
        doc_ref = self.db.collection("VideoAnalysis").document()
        formatted_data = {
            f"fragment{idx + 1}": result_dict
            for idx, result_dict in enumerate(data)
        }
        doc_ref.set(formatted_data)
        return doc_ref.id
