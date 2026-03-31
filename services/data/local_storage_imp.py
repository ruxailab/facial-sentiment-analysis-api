import os
import shutil

class LocalStorageImp:
    def __init__(self, base_path="local_storage"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def download_video_from_storage(self, video_name):
        local_file = os.path.join(self.base_path, video_name)
        return local_file