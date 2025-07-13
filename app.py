import cv2
import mediapipe as mp
from google.cloud import storage, secretmanager
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import datetime
import logging
import subprocess
import firebase_admin
from firebase_admin import credentials, firestore
import json
from exercise_analysis import SquatAnalyzer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://*.argus.fit", "https://www.argus.fit", "https://blo-upload"]}})
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if not firebase_admin._apps:
    cred = firebase_admin.credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
db = firestore.client()

def get_service_account_key():
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/zeta-dock-388018/secrets/flask-app-service-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    secret_payload = response.payload.data.decode("UTF-8")
    return service_account.Credentials.from_service_account_info(eval(secret_payload))

def download_video_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Downloaded {source_blob_name} to {destination_file_name}.")

def fetch_metadata_from_firestore(video_id):
    doc_ref = db.collection("userVideos").document(video_id)
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else None

def analyze_video(video_path, metadata, output_path):
    logging.info(f"Analyzing video: {video_path} with metadata: {metadata}")
    exercise_analyzer = SquatAnalyzer()
    
    if user_load := metadata.get('load'):
        try:
            exercise_analyzer.set_user_weight(float(user_load))
        except (ValueError, TypeError):
            exercise_analyzer.set_user_weight(1.0)

    needs_rotation = metadata.get('isPortrait', False)
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9, model_complexity=2)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dims = (frame_height, frame_width) if needs_rotation else (frame_width, frame_height)
    out = cv2.VideoWriter(output_path, fourcc, fps, out_dims)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        if needs_rotation: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_world_landmarks:
            exercise_analyzer.analyze_frame(results, frame_count, fps, frame)
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # **** NEW LOGIC: SEPARATE SUMMARY FROM TIME SERIES ****
    summary_results, time_series_data = exercise_analyzer.get_final_analysis()

    if not summary_results:
        logging.error("Analysis produced no summary results.")
        return False
        
    video_id = metadata.get('videoName', '').split('.')[0]
    if not video_id:
        logging.error("Could not determine video ID.")
        return False

    # Convert numpy types for summary
    firestore_summary = exercise_analyzer.convert_numpy_types(summary_results)
    
    # Save summary to main document
    doc_ref = db.collection("exerciseAnalysis").document(video_id)
    doc_ref.set(firestore_summary)
    logging.info(f"Successfully saved summary analysis to Firestore with ID: {video_id}")

    # Save time series to subcollection in a batch
    batch = db.batch()
    time_series_ref = doc_ref.collection('timeSeries')
    for frame_data in time_series_data:
        frame_doc_ref = time_series_ref.document(str(frame_data['frame_index']))
        converted_frame_data = exercise_analyzer.convert_numpy_types(frame_data)
        batch.set(frame_doc_ref, converted_frame_data)
    batch.commit()
    logging.info(f"Successfully saved {len(time_series_data)} frames to timeSeries subcollection.")

    return True

@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        event_data = request.get_json()
        if not event_data or not (bucket_name := event_data.get("bucket")) or not (video_name := event_data.get("name")):
            return "Bad Request: Missing bucket or object name", 400
        
        video_id = video_name.rsplit(".", 1)[0]
        metadata = fetch_metadata_from_firestore(video_id)
        if not metadata:
            return f"No metadata for {video_id}", 404

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
            input_path = temp_input_file.name
            output_path = temp_output_file.name

        try:
            download_video_from_gcs(bucket_name, video_name, input_path)
            if not analyze_video(input_path, metadata, output_path):
                raise Exception("Video analysis failed")
            
            # This part for uploading the analyzed video can be re-enabled if needed
            # upload_analyzed_video_to_gcs(output_path, f"{video_id}_analyzed.mp4")

        finally:
            if os.path.exists(input_path): os.remove(input_path)
            if os.path.exists(output_path): os.remove(output_path)
            
        return "Video processed successfully", 200

    except Exception as e:
        logging.error(f"Error processing video: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route('/exercise-analysis/<video_id>', methods=['GET'])
def get_exercise_analysis(video_id):
    try:
        # **** NEW LOGIC: READ FROM DOCUMENT AND SUBCOLLECTION ****
        doc_ref = db.collection("exerciseAnalysis").document(video_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({'status': 'error', 'message': 'Analysis not found'}), 404
        
        analysis_data = doc.to_dict()
        
        # Fetch time series data from subcollection
        time_series_ref = doc_ref.collection('timeSeries').order_by('timestamp').stream()
        time_series_list = [frame.to_dict() for frame in time_series_ref]
        
        analysis_data['time_series'] = time_series_list
        
        return jsonify(analysis_data)
            
    except Exception as e:
        logging.error(f"Error fetching exercise analysis: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# Keep your other endpoints as they are.

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)