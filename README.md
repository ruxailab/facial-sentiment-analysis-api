# RuxaiLab Facial Emotion Detection API

## Overview
This Flask API serves as the backend for the RuxaiLab Bachelor's thesis project, providing facial emotion detection capabilities. It processes video input, detects emotions displayed on faces within the video frames, and integrates with Firebase.

## Features
- **Video Processing:** Upload a video file to analyze emotions displayed on faces within the frames.
- **Facial Emotion Detection:** Utilizes a pre-trained model to analyze emotions in video frames.
- **Firebase Integration:** User authentication and storage are managed through Firebase, ensuring secure access to the API.
- **Logging:** Utilizes Python's logging module for informational and error messages.
  
## Environment / Versions
- Python: 3.12
- Poetry: 1.8.x
- TensorFlow (CPU): 2.18.0
- Keras: 3.3.3
- OpenCV: 4.9.0
- Flask: 3.0.3
- Firebase Admin SDK: 6.5.0
  
## Project Structure
```
backend/
│ app.py
│ model_loader.py
│ firebase_service.py
│ video_processor.py
│ requirements.txt (gerado só se necessário)
│ .env
└─ models/
```

## Setup
#### 1. Clone the repository
```
git clone https://github.com/your-username/ruxailab-facial-emotion-api.git
```

#### 2. Install Poetry (if not installed)
https://python-poetry.org/docs/#installation

#### 3. Install dependencies
```
poetry install
```

### 4. Firebase Configuration (Required)

---

##### 4.1 Storage Bucket
1. Open **Firebase Console → Storage**  
2. Copy your bucket name (e.g. `project-id.appspot.com`)  
3. Add it to your `.env` file:
```
STORAGE_BUCKET=your-bucket-here
```

##### 4.2 Admin SDK Credentials
1. Go to **Firebase Console → Project Settings → Service Accounts**  
2. Select **Python**  
3. Click **Generate new private key**  
4. Download the JSON file  
5. Move it to the project root and rename it to:
```
service-account.json
```

 This file is required for all Firebase Admin SDK operations.

---

#### 5. Start the API
```
poetry run python -u app.py
```

## Usage
### Uploading a Video
- Send a POST request to `/process_video` endpoint with the name of the video file to be analyzed in the request body.
- The API will download the video from Firebase Storage, perform emotion analysis, and store the results in Firestore.
- Once the API is running, it automatically detects new videos uploaded to Firebase Storage, processes them to analyze emotions, and uploads the results to Firestore.

### Testing Firebase Function
- Send a GET request to `/test` endpoint.
- The API will call a Firebase function hosted elsewhere for testing purposes.

## API Routes
- **POST /process_video:** Initiates emotion analysis on the uploaded video.
- **GET /test:** Calls a Firebase function for testing purposes.

## Deployment

The API is deployed as a **CPU-only** Dockerized Flask service on **Google Cloud Run**. Shared deploy logic lives in [`.github/workflows/deploy-reusable.yml`](.github/workflows/deploy-reusable.yml); callers trigger it per branch:

| Branch | Workflow | GCP project | Cloud Run service |
|---|---|---|---|
| `main` | [`deploy-prod.yml`](.github/workflows/deploy-prod.yml) | `ruxailab-prod` | `facial-sentiment-analysis-api` |
| `develop` | [`deploy-dev.yml`](.github/workflows/deploy-dev.yml) | `ruxailab-develop` | `facial-sentiment-analysis-api` |

| Setting | Value |
|---|---|
| Region | `us-central1` |
| Artifact Registry repo | `containers` |
| Image | `facial-sentiment-analysis-api` |
| Resources | 2 CPU · 4 Gi memory · port 8080 · CPU boost |

### Prerequisites

- Google Cloud project with billing enabled (`ruxailab-prod` / `ruxailab-develop`)
- Artifact Registry repository `containers` in `us-central1`
- APIs enabled: Artifact Registry, Cloud Run
- GitHub repository secrets:
  - **`GCP_SA_KEY_PROD`** / **`GCP_SA_KEY_DEV`** — JSON keys for each environment's GCP service account
  - **`FIREBASE_STORAGE_BUCKET_PROD`** / **`FIREBASE_STORAGE_BUCKET_DEV`** — Firebase Storage bucket names

Each caller maps its environment-specific secrets to the generic names expected by the reusable workflow (`GCP_SA_KEY`, `FIREBASE_STORAGE_BUCKET`). The service account needs Artifact Registry Writer, Cloud Run Admin, and Service Account User.

### Automatic deploy

1. Configure the secrets above in the GitHub repository settings.
2. Push (or merge) to `main` (prod) or `develop` (dev), with changes under `app.py`, `routes/**`, `services/**`, `utils/**`, `schemas/**`, `models/**`, `.github/workflows/**`, `Dockerfile`, `pyproject.toml`, or `poetry.lock`.
3. The workflow will:
   - Authenticate to GCP with `GCP_SA_KEY_PROD` or `GCP_SA_KEY_DEV`
   - Build and push `us-central1-docker.pkg.dev/<project>/containers/facial-sentiment-analysis-api:sha-<short-sha>`
   - Deploy the image to Cloud Run (with `FIREBASE_STORAGE_BUCKET`), then delete the Artifact Registry image tag

