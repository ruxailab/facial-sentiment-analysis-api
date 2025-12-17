from abc import ABC, abstractmethod

class FirebaseService(ABC):
    @abstractmethod
    def download_video_from_storage(self, video_name: str) -> str:
        pass
    @abstractmethod
    def upload_to_firestore(self, data) -> str:
        pass