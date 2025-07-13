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

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://*.argus.fit", "https://www.argus.fit", "https://blo-upload"]}})

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:  # Prevent reinitialization during hot-reloads
    cred = firebase_admin.credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

def get_service_account_key():
    # Create a client to interact with Google Secret Manager
    client = secretmanager.SecretManagerServiceClient()

    # Set the resource name for the secret
    secret_name = "flask-app-service-key"
    project_id = "zeta-dock-388018"
    secret_version = "latest"  # You can also use specific versions if needed

    # Create the resource path for the secret
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{secret_version}"

    # Access the secret version
    response = client.access_secret_version(request={"name": name})

    # Extract the secret payload and decode it
    secret_payload = response.payload.data.decode("UTF-8")

    # Create credentials object from the secret JSON
    return service_account.Credentials.from_service_account_info(eval(secret_payload))

def get_storage_client_with_credentials():
    # Initialize Google Cloud Storage client with service account credentials
    credentials = get_service_account_key()
    return storage.Client(credentials=credentials)

def download_video_from_gcs(bucket_name, source_blob_name, destination_file_name):
    # Downloads a video file from Google Cloud Storage.
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def fetch_metadata_from_firestore(video_id):
    #Fetch metadata from Firestore based on the video ID.
    doc_ref = db.collection("userVideos").document(video_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        logging.error(f"No metadata found for video ID: {video_id}")
        return None
    
def get_video_rotation(input_path):
    """Get video rotation from multiple metadata sources"""
    try:
        # Check rotation in stream metadata (container level)
        cmd_stream = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        rotation_stream = subprocess.check_output(cmd_stream).decode('utf-8').strip()
        logging.info(f"Detected rotation from stream tags: {rotation_stream}")
        
        if rotation_stream:
            return int(rotation_stream)

        # Check rotation in display matrix
        cmd_matrix = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=side_data_list',
            '-of', 'json',
            input_path
        ]
        matrix_output = json.loads(subprocess.check_output(cmd_matrix).decode('utf-8'))
        logging.info(f"Display matrix output: {matrix_output}")
        
        if 'streams' in matrix_output and matrix_output['streams']:
            side_data_list = matrix_output['streams'][0].get('side_data_list', [])
            for side_data in side_data_list:
                if 'rotation' in side_data:
                    logging.info(f"Found rotation in display matrix: {side_data['rotation']}")
                    return int(side_data['rotation'])

        logging.info("No rotation metadata found in any source")
        return 0

    except Exception as e:
        logging.error(f"Error reading rotation metadata: {str(e)}")
        return 0
    
def debug_video_metadata(input_path):
    """Print all available metadata for debugging"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_format',
            '-show_streams',
            '-of', 'json',
            input_path
        ]
        metadata = json.loads(subprocess.check_output(cmd).decode('utf-8'))
        logging.info("Full video metadata:")
        logging.info(json.dumps(metadata, indent=2))
    except Exception as e:
        logging.error(f"Error getting debug metadata: {str(e)}")

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for Firestore compatibility"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def analyze_video(video_path, metadata, output_path):
    """
    Analyze the video using MediaPipe's 3D pose estimation, draw landmarks, and save the analyzed video.
    """
    logging.info(f"Analyzing video: {video_path} with metadata: {metadata}")

    # Initialize exercise analyzer
    exercise_analyzer = SquatAnalyzer()
    
    # Set user weight from metadata if available
    user_load = metadata.get('load')
    if user_load:
        try:
            # Convert to float and set
            load_kg = float(user_load)
            exercise_analyzer.set_user_weight(load_kg)
            logging.info(f"Set user load to {load_kg} kg for volume calculations")
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not parse user load '{user_load}': {str(e)}")
            # Default to 1 kg if parsing fails
            exercise_analyzer.set_user_weight(1.0)

    # Get rotation from metadata
    needs_rotation = metadata.get('isPortrait', False)
    logging.info(f"Video needs rotation based on metadata: {needs_rotation}")

    # Initialize MediaPipe pose solution with 3D tracking
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        model_complexity=2,  # Use the most accurate model
        enable_segmentation=False,
        smooth_landmarks=True
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return False

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
        logging.warning(f"Invalid FPS detected, defaulting to {fps}")

    # Create temporary file for initial OpenCV processing
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    
    # Initialize video writer with adjusted dimensions if video is rotated
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if needs_rotation:
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            fps,
            (frame_height, frame_width),
            isColor=True
        )
    else:
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            fps,
            (frame_width, frame_height),
            isColor=True
        )

    if not out.isOpened():
        logging.error("Failed to initialize VideoWriter")
        cap.release()
        return False

    frame_count = 0
    landmarks_detected = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Rotate frame if needed
            if needs_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw landmarks with 3D visualization - simplified to just show specific landmarks
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                landmarks_detected += 1
                
                # Process frame with the exercise analyzer - this now draws only selected landmarks
                frame_metrics = exercise_analyzer.analyze_frame(
                    results.pose_landmarks,
                    frame_count,
                    fps,
                    frame
                )
            
            # Write the frame
            out.write(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                logging.info(f"Processed {frame_count} frames, detected landmarks in {landmarks_detected} frames")
            
        # Get final analysis after processing all frames
        analysis_results = exercise_analyzer.get_final_analysis()

        logging.info(f"Final analysis results: \n{json.dumps(analysis_results, indent=2)}")


        # Extract video ID with logging
        original_video_name = metadata.get('videoName')
        video_id = original_video_name.split('.')[0] if original_video_name else video_path.split('/')[-1].split('.')[0]
        logging.info(f"Original video name from metadata: {original_video_name}")
        logging.info(f"Using video ID for Firestore: {video_id}")
        
        # Save analysis results to Firestore
        try:
            # Convert NumPy types to native Python types for Firestore compatibility
            firestore_compatible_results = convert_numpy_types(analysis_results)
            db.collection("exerciseAnalysis").document(video_id).set(firestore_compatible_results)
            logging.info(f"Successfully saved analysis results to Firestore with ID: {video_id}")
        except Exception as e:
            logging.error(f"Error saving to Firestore. Video ID: {video_id}, Error: {str(e)}")
            raise

        logging.info(f"Exercise analysis completed. Results: {analysis_results}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Use ffmpeg for final encoding
    try:
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        logging.info(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        logging.info("Successfully transcoded video with ffmpeg")
        
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
    except Exception as e:
        logging.error(f"Error during ffmpeg processing: {str(e)}")
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        logging.info(f"Successfully created video: {output_path}")
        return True
    else:
        logging.error("Failed to create valid output video")
        return False
            
def upload_analyzed_video_to_gcs(source_file_path, destination_blob_name):
    client = storage.Client()
    output_bucket_name = "gym-videos-out"
    bucket = client.bucket(output_bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)
    logging.info(f"Analyzed file uploaded to GCS: gs://{output_bucket_name}/{destination_blob_name}")
    return destination_blob_name

@app.route('/test-firebase', methods=['GET'])
def test_firebase():
    try:
        test_data = {"message": "Hello from Firebase!"}
        db.collection("testCollection").add(test_data)
        return {"message": "Firebase is working!"}, 200
    
    except Exception as e:
        logging.error(f"Error in test_firebase: {e}", exc_info=True)
        return {"error": str(e)}, 500


@app.route('/generate-signed-url', methods=['POST', 'OPTIONS'])
def generate_signed_url():
    # Generate a signed URL for uploading a file to Google Cloud Storage.
    try:
        if request.method == 'OPTIONS':
            # Handle CORS preflight
            response = jsonify({'message': 'Preflight OK'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response

        # Handle POST request
        request_data = request.get_json()
        file_name = request_data.get('file_name')

        if not file_name:
            return jsonify({'error': 'File name is required'}), 400

        storage_client = get_storage_client_with_credentials()
        
        bucket_name = "gym-videos-in"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Generate signed URL with the required content type
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="PUT",
            content_type="video/mp4"  # Enforce the content type
        )

        return jsonify({'url': signed_url})

    except Exception as e:
        logging.error(f"Error in generate_signed_url: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        logging.info("Received request to process video.")

        event_data = request.get_json()
        if not event_data:
            return "Bad Request: No event data received", 400

        # Extract event ID and check for duplicate processing
        event_id = event_data.get("id", "")
        if event_id:
            doc_ref = db.collection("processed_events").document(event_id)
            doc = doc_ref.get()
            if doc.exists:
                logging.info(f"Event {event_id} already processed, skipping")
                return jsonify({"message": "Event already processed"}), 200
            doc_ref.set({"processed_at": datetime.datetime.utcnow().isoformat()})

        bucket_name = event_data.get("bucket", "")
        video_name = event_data.get("name", "")

        if not bucket_name or not video_name:
            logging.error("Missing bucket or object name in Eventarc event.")
            return "Bad Request: Missing bucket or object name", 400

        logging.info(f"Processing video from bucket: {bucket_name}, file: {video_name}")

        video_id = video_name.rsplit(".", 1)[0]
        analyzed_video_name = f"{video_id}_analyzed.mp4"

        # Check if analyzed video already exists
        storage_client = get_storage_client_with_credentials()
        analyzed_bucket = storage_client.bucket("gym-videos-out")
        analyzed_blob = analyzed_bucket.blob(analyzed_video_name)

        if analyzed_blob.exists():
            logging.info(f"Skipping already processed video: {analyzed_video_name}")
            return jsonify({"message": "Video already processed"}), 200

        # Fetch metadata
        metadata = fetch_metadata_from_firestore(video_id)
        if not metadata:
            logging.error(f"No metadata found for video ID: {video_id}")
            return jsonify({"error": f"No metadata found for video ID: {video_id}"}), 200

        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
            input_path = temp_input_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
            output_path = temp_output_file.name

        try:
            # Download input video
            download_video_from_gcs(bucket_name, video_name, input_path)
            logging.info("Video downloaded successfully")

            # Process video
            result = analyze_video(input_path, metadata, output_path)
            if not result:
                raise Exception("Video analysis failed")

            # Upload processed video
            upload_analyzed_video_to_gcs(output_path, analyzed_video_name)
            
            return jsonify({
                "message": "Video processed successfully",
                "analyzedVideo": analyzed_video_name
            }), 200

        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            logging.info("Temporary files cleaned up")

    except Exception as e:
        error_message = "Internal Server Error during video processing"
        status_code = 500
        if isinstance(e, NotFound):
            error_message = f"Resource not found: {e}"
            status_code = 404
        # Note: KeyError and ValueError handling was removed as per the unified diff request
        elif isinstance(e, ValueError):
             error_message = f"Invalid data in event payload: {e}"
             status_code = 400
        else:
            # Log unhandled exceptions with full traceback
            logging.error(f"Unhandled error processing video: {e}", exc_info=True)

        return jsonify({"error": error_message}), status_code

@app.route('/exercise-analysis/<video_id>', methods=['GET'])
def get_exercise_analysis(video_id):
    try:
        logging.info(f"Fetching exercise analysis for video_id: {video_id}")
        
        # Fetch analysis results from Firestore
        doc_ref = db.collection("exerciseAnalysis").document(video_id)
        doc = doc_ref.get()
        
        if doc.exists:
            logging.info(f"Analysis found for video_id: {video_id}")
            analysis_data = doc.to_dict()
            logging.info(f"Analysis data: {analysis_data}")
            return jsonify(analysis_data)
        else:
            logging.warning(f"No analysis found for video_id: {video_id}")
            return jsonify({
                'status': 'error',
                'message': 'Analysis not found'
            }), 404
            
    except Exception as e:
        logging.error(f"Error fetching exercise analysis: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/save-video-info', methods=['POST'])
def save_video_info():
    try:
        data = request.get_json()
        logging.info(f"Received metadata: {data}")

        video_name = data.get("videoName")
        video_id = video_name.rsplit(".", 1)[0]

        # Save metadata to Firestore with videoID as the document ID
        db.collection("userVideos").document(video_id).set({
            "email": data.get("email"),
            "weight": data.get("weight"),
            "height": data.get("height"),
            "load": data.get("load"),
            "videoName": video_name,
            "isPortrait": data.get("isPortrait", False),  # Add this line
            "uploadedAt": datetime.datetime.utcnow().isoformat(),
        })

        logging.info(f"Metadata saved successfully with videoID: {video_id}")
        return jsonify({"message": "Video info saved successfully"}), 200
    
    except Exception as e:
        logging.error(f"Error saving video info: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500
    
@app.route('/video-status/<video_name>', methods=['GET'])
def get_video_status(video_name):
    try:
        logging.info(f"Checking status for video: {video_name}")

        credentials = get_service_account_key()
        storage_client = storage.Client(credentials=credentials)
        
        analyzed_video_name = f"{video_name.rsplit('.', 1)[0]}_analyzed.mp4"
        
        # Check output bucket
        output_bucket = storage_client.bucket('gym-videos-out')
        output_blob = output_bucket.blob(analyzed_video_name)
        
        if output_blob.exists():
            # Generate a signed URL that's valid for 1 hour
            signed_url = output_blob.generate_signed_url(
                version='v4',
                expiration=datetime.timedelta(hours=1),
                method='GET'
            )
            return jsonify({
                'status': 'complete',
                'processed_url': signed_url
            })
        
        # Check if original video exists in input bucket
        input_bucket = storage_client.bucket('gym-videos-in')
        input_blob = input_bucket.blob(video_name)
        
        if input_blob.exists():
            return jsonify({'status': 'processing'})
        
        return jsonify({
            'status': 'not found',
            'message': 'Video not found in either bucket'
        }), 404

    except Exception as e:
        logging.error(f"Error checking video status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error checking video status'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
