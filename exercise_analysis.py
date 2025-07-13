import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
import logging
from collections import deque

class ExerciseAnalyzer:
    """Base class for exercise analysis using 3D world landmarks."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Store previous frame's data for kinetic calculations
        self.prev_world_landmarks = None
        self.prev_velocities = {}

        # --- Define all landmarks we want to track ---
        self.tracked_landmarks = [
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
            mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
            mp.solutions.pose.PoseLandmark.LEFT_THUMB,
            mp.solutions.pose.PoseLandmark.RIGHT_THUMB
        ]

    def get_3d_point(self, landmark) -> np.ndarray:
        """Convert MediaPipe landmark to 3D numpy array."""
        return np.array([landmark.x, landmark.y, landmark.z])

    def calculate_3d_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three 3D points robustly."""
        v1 = p1 - p2
        v2 = p3 - p2

        # **** ROBUSTNESS FIX ****
        # Check for zero vectors to prevent division by zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0 # Or np.nan, but 0.0 is safer for JSON

        v1_norm = v1 / norm_v1
        v2_norm = v2 / norm_v2
        
        # Calculate angle using dot product, clipping to avoid math errors
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

    def calculate_all_joint_angles(self, world_landmarks) -> Dict[str, float]:
        """Calculate all relevant joint angles from world landmarks."""
        points = {lm: self.get_3d_point(world_landmarks.landmark[lm]) for lm in self.tracked_landmarks}
        angles = {}
        # Knees
        angles['left_knee'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.LEFT_HIP], points[self.mp_pose.PoseLandmark.LEFT_KNEE], points[self.mp_pose.PoseLandmark.LEFT_ANKLE])
        angles['right_knee'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.RIGHT_HIP], points[self.mp_pose.PoseLandmark.RIGHT_KNEE], points[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
        # Hips
        angles['left_hip'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.LEFT_SHOULDER], points[self.mp_pose.PoseLandmark.LEFT_HIP], points[self.mp_pose.PoseLandmark.LEFT_KNEE])
        angles['right_hip'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], points[self.mp_pose.PoseLandmark.RIGHT_HIP], points[self.mp_pose.PoseLandmark.RIGHT_KNEE])
        # Shoulders
        angles['left_shoulder'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.LEFT_ELBOW], points[self.mp_pose.PoseLandmark.LEFT_SHOULDER], points[self.mp_pose.PoseLandmark.LEFT_HIP])
        angles['right_shoulder'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.RIGHT_ELBOW], points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], points[self.mp_pose.PoseLandmark.RIGHT_HIP])
        # Elbows
        angles['left_elbow'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.LEFT_SHOULDER], points[self.mp_pose.PoseLandmark.LEFT_ELBOW], points[self.mp_pose.PoseLandmark.LEFT_WRIST])
        angles['right_elbow'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], points[self.mp_pose.PoseLandmark.RIGHT_ELBOW], points[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        # Ankles
        angles['left_ankle'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.LEFT_KNEE], points[self.mp_pose.PoseLandmark.LEFT_ANKLE], points[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
        angles['right_ankle'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.RIGHT_KNEE], points[self.mp_pose.PoseLandmark.RIGHT_ANKLE], points[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
        # Wrists
        angles['left_wrist'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.LEFT_ELBOW], points[self.mp_pose.PoseLandmark.LEFT_WRIST], points[self.mp_pose.PoseLandmark.LEFT_THUMB])
        angles['right_wrist'] = self.calculate_3d_angle(points[self.mp_pose.PoseLandmark.RIGHT_ELBOW], points[self.mp_pose.PoseLandmark.RIGHT_WRIST], points[self.mp_pose.PoseLandmark.RIGHT_THUMB])
        return angles

    def calculate_kinetics(self, world_landmarks, fps: float) -> Tuple[Dict, Dict]:
        """Calculate real-world velocities and accelerations for all joints."""
        velocities = {}
        accelerations = {}
        if self.prev_world_landmarks:
            for lm in self.tracked_landmarks:
                current_pos = self.get_3d_point(world_landmarks.landmark[lm])
                prev_pos = self.get_3d_point(self.prev_world_landmarks.landmark[lm])
                velocity = (current_pos - prev_pos) * fps
                velocities[lm.name] = velocity
                if self.prev_velocities and lm.name in self.prev_velocities:
                    prev_velocity = self.prev_velocities[lm.name]
                    acceleration = (velocity - prev_velocity) * fps
                    accelerations[lm.name] = acceleration
        return velocities, accelerations

class SquatAnalyzer(ExerciseAnalyzer):
    """Analyzer specific to squat exercises, using world landmarks."""
    def __init__(self):
        super().__init__()
        # Initialize all necessary variables for analysis
        self.frame_metrics = []
        self.user_weight = 0
        self.exercise_state = "inactive"
        # Add back other state variables as you rebuild the logic
        self.reps_completed = 0
        self.rep_state = "standing"

    def set_user_weight(self, load_kg: float):
        """Set the user's loaded weight for volume calculations"""
        self.user_weight = load_kg

    def analyze_frame(self, pose_results, frame_idx: int, fps: float, frame) -> Optional[Dict]:
        """
        Analyze a single frame using world landmarks for kinetics and angles,
        and image landmarks for drawing.
        """
        if not pose_results.pose_world_landmarks or not pose_results.pose_landmarks:
            self.prev_world_landmarks = None
            self.prev_velocities = {}
            return None

        world_landmarks = pose_results.pose_world_landmarks
        image_landmarks = pose_results.pose_landmarks
        
        joint_angles = self.calculate_all_joint_angles(world_landmarks)
        velocities, accelerations = self.calculate_kinetics(world_landmarks, fps)

        frame_data = {
            'frame_index': frame_idx,
            'timestamp': frame_idx / fps,
            'joint_angles': joint_angles,
            'velocities': {name: vel.tolist() for name, vel in velocities.items()},
            'accelerations': {name: acc.tolist() for name, acc in accelerations.items()}
        }
        
        self.frame_metrics.append(frame_data)

        # Update state for the next frame
        self.prev_world_landmarks = world_landmarks
        self.prev_velocities = velocities

        # Draw the landmarks on the frame using image_landmarks
        self.draw_landmarks_with_state(frame, image_landmarks, self.exercise_state, {})
        
        return frame_data

    def draw_landmarks_with_state(self, frame, landmarks, exercise_state: str, rep_info: Dict):
        """Draw landmarks with different colors based on exercise state"""
        # Using the standard mediapipe drawing utils for simplicity and robustness
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def get_final_analysis(self) -> Dict:
        """Get the final analysis with the complete time series data."""
        if not self.frame_metrics:
            return {'status': 'error', 'message': 'No frames were analyzed.'}

        # The core of the final result is the detailed time_series data.
        # You can add back your higher-level scoring (Volume, TUT, etc.) here
        # by processing the data in self.frame_metrics.
        analysis_results = {
            'status': 'success',
            'rep_counting': {
                'completed_reps': self.reps_completed, # Placeholder
            },
            'metrics': {
                'total_frames_analyzed': len(self.frame_metrics),
            },
            'time_series': self.frame_metrics
        }
        
        return self.convert_numpy_types(analysis_results)

    def convert_numpy_types(self, obj):
        """Recursively converts numpy types to native Python types for Firestore compatibility."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj