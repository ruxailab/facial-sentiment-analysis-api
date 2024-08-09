import firebase_admin
import logging
import coloredlogs
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from services.data.firebase_service import FirebaseService

class FirebaseImp(FirebaseService):
    
    coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    def __init__(self, firebase_cred_path, storage_bucket):
        self.firebase_cred_path = firebase_cred_path
        self.storage_bucket = storage_bucket
        self._initialize_app()
        coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.db = firestore.client()

    def _initialize_app(self):
        cred = credentials.Certificate(self.firebase_cred_path)
        firebase_admin.initialize_app(cred, {'storageBucket': self.storage_bucket})
        self.storage_client = storage.bucket()

    def download_video_from_storage(self,video_name):
        self.logger.info(f"Attempting to download video: {video_name} from storage.")
        blob = self.storage_client.blob(video_name)
        video_path = f"static/videos/{video_name}"
        blob.download_to_filename(f"{video_path}")
        self.logger.info(f"Video downloaded successfully to: {video_path}")
        return video_path

    def upload_to_firestore(self, data):
        doc_ref = self.db.collection('VideoAnalysis').document()
        formatted_data = {f"fragment{idx + 1}": result_dict for idx, result_dict in enumerate(data)}
        doc_ref.set(formatted_data)
        return doc_ref.id
