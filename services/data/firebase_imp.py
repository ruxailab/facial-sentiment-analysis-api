import firebase_admin
import logging
import coloredlogs

from firebase_admin import firestore, storage
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
        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                options={
                    "storageBucket": self.storage_bucket
                }
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
