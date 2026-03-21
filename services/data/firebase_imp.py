import firebase_admin
from firebase_admin import firestore, storage
from services.data.firebase_service import FirebaseService
from utils.logger import get_logger

logger = get_logger(__name__)


class FirebaseImp(FirebaseService):

    def __init__(self, storage_bucket: str):
        self.storage_bucket = storage_bucket
        self._initialize_app()
        self.db = firestore.client()
        self.storage_client = storage.bucket()

    def _initialize_app(self):
        if not firebase_admin._apps:
            logger.info("Initializing Firebase app with bucket: %s", self.storage_bucket)
            firebase_admin.initialize_app(
                options={
                    "storageBucket": self.storage_bucket
                }
            )
        else:
            logger.debug("Firebase app already initialized, skipping.")

    def download_video_from_storage(self, video_name: str) -> str:
        logger.info("Attempting to download video: %s from storage.", video_name)
        blob = self.storage_client.blob(video_name)
        video_path = f"static/videos/{video_name}"
        blob.download_to_filename(video_path)
        logger.info("Video downloaded successfully to: %s", video_path)
        return video_path

    def upload_to_firestore(self, data) -> str:
        doc_ref = self.db.collection("VideoAnalysis").document()
        formatted_data = {
            f"fragment{idx + 1}": result_dict
            for idx, result_dict in enumerate(data)
        }
        doc_ref.set(formatted_data)
        logger.info("Uploaded analysis results to Firestore document: %s", doc_ref.id)
        return doc_ref.id
