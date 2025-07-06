import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
import logging

class ExerciseAnalyzer:
    """Base class for exercise analysis with 3D pose normalization"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def get_3d_point(self, landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z])

    def get_body_coordinate_system(self, landmarks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        left_shoulder = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])

        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        y_axis = shoulder_center - hip_center
        y_axis /= np.linalg.norm(y_axis)

        hip_line = right_hip - left_hip
        z_axis = np.cross(y_axis, hip_line)
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        return y_axis, z_axis, x_axis

    def normalize_pose(self, landmarks) -> Dict[str, np.ndarray]:
        left_hip_3d = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_3d = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center = (left_hip_3d + right_hip_3d) / 2

        y_axis, z_axis, x_axis = self.get_body_coordinate_system(landmarks)
        normalized_points = {}
        landmark_list = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        for landmark_idx in landmark_list:
            point_3d = self.get_3d_point(landmarks.landmark[landmark_idx])
            centered_point = point_3d - hip_center
            new_x = np.dot(centered_point, x_axis)
            new_y = np.dot(centered_point, y_axis)
            new_z = np.dot(centered_point, z_axis)
            normalized_points[landmark_idx] = np.array([new_x, new_y, new_z])

        return normalized_points

    def calculate_3d_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_joint_angles(self, normalized_points: Dict[str, np.ndarray]) -> Dict[str, float]:
        angles = {}
        angles['left_knee'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_KNEE],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        )
        angles['right_knee'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_KNEE],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        )
        angles['left_hip'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_KNEE]
        )
        angles['right_hip'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        )
        return angles

class SquatAnalyzer(ExerciseAnalyzer):
    """
    Analyzes squats by first detecting the start and end of the exercise period,
    then performing a detailed analysis only on the relevant frames.
    """
    def __init__(self):
        super().__init__()
        
        # State machine: 'searching', 'analyzing', 'finished'
        self.analysis_state = "searching"
        self.exercise_start_frame = None
        self.inactive_frame_counter = 0
        self.INACTIVE_DURATION_THRESHOLD = 90  # frames (e.g., 3 seconds at 30fps)
        self.START_VELOCITY_THRESHOLD = -0.1   # Velocity to detect the first descent

        # Rep counting
        self.rep_state = "standing"
        self.reps_completed = 0
        self.rep_details = []
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.start_hip_height = None

        # Metrics
        self.frame_metrics = []
        self.total_volume = 0
        self.user_weight = 0
        self.max_acceleration = 0
        self.avg_acceleration = []
        
        # Previous state variables
        self.prev_hip_height = None
        self.prev_hip_velocity = 0
        
    def set_user_weight(self, load_kg: float):
        self.user_weight = load_kg
        
    def analyze_frame(self, landmarks, frame_idx: int, fps: float, frame) -> Dict:
        if not landmarks or self.analysis_state == "finished":
            return None

        # --- Always calculate basic kinematics ---
        left_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center = (left_hip + right_hip) / 2
        hip_height = hip_center[1]

        hip_velocity = (hip_height - self.prev_hip_height) * fps if self.prev_hip_height is not None else 0

        # --- State Machine Logic ---
        if self.analysis_state == "searching":
            # Look for the first downward movement
            if hip_velocity < self.START_VELOCITY_THRESHOLD:
                self.analysis_state = "analyzing"
                self.exercise_start_frame = frame_idx
                logging.info(f"Exercise start detected at frame {frame_idx}")
        
        elif self.analysis_state == "analyzing":
            # Perform the detailed analysis for this frame
            self.process_active_frame(landmarks, frame_idx, fps, hip_height, hip_velocity)

            # Check if the exercise has ended
            if abs(hip_velocity) < 0.05: # Check for very low movement
                self.inactive_frame_counter += 1
            else:
                self.inactive_frame_counter = 0 # Reset counter if movement is detected

            if self.inactive_frame_counter >= self.INACTIVE_DURATION_THRESHOLD:
                self.analysis_state = "finished"
                logging.info(f"Exercise end detected at frame {frame_idx}")

        # --- Update previous values for the next frame ---
        self.prev_hip_height = hip_height
        self.prev_hip_velocity = hip_velocity
        
        # Draw landmarks on the frame
        self.draw_landmarks(frame, landmarks, self.analysis_state)

        return {'analysis_state': self.analysis_state}

    def process_active_frame(self, landmarks, frame_idx: int, fps: float, hip_height: float, hip_velocity: float):
        """Processes a single frame during the 'analyzing' state."""
        normalized_points = self.normalize_pose(landmarks)
        angles = self.calculate_joint_angles(normalized_points)
        avg_knee_angle = (angles['left_knee'] + angles['right_knee']) / 2
        
        self.count_reps(hip_height, hip_velocity, frame_idx, fps)
        
        is_concentric = hip_height > self.prev_hip_height
        hip_acceleration = (hip_velocity - self.prev_hip_velocity) * fps
        
        intensity = abs(hip_acceleration) if is_concentric else 1.0 / (1.0 + abs(hip_acceleration))
        self.avg_acceleration.append(intensity)
        self.max_acceleration = max(self.max_acceleration, intensity)
        
        frame_volume = 0
        if is_concentric and self.user_weight > 0:
            vertical_distance = abs(hip_height - self.prev_hip_height)
            frame_volume = vertical_distance * self.user_weight
            self.total_volume += frame_volume
        
        self.frame_metrics.append({
            'time': frame_idx / fps, 'avg_knee_angle': avg_knee_angle, 'hip_height': hip_height,
            'hip_velocity': hip_velocity, 'hip_acceleration': hip_acceleration, 'is_concentric': is_concentric,
            'phase_intensity': intensity, 'accumulated_volume': self.total_volume,
            'rep_state': self.rep_state, 'current_reps': self.reps_completed
        })

    def count_reps(self, hip_height: float, hip_velocity: float, frame_idx: int, fps: float):
        current_time = frame_idx / fps
        if self.rep_state == "standing":
            if hip_velocity < -0.05:
                self.rep_state = "descending"
                self.current_rep_start_frame = frame_idx
                self.current_rep_start_time = current_time
                self.start_hip_height = hip_height
        elif self.rep_state == "descending":
            if hip_velocity >= 0:
                self.rep_state = "ascending"
        elif self.rep_state == "ascending":
            is_at_top = hip_height >= self.start_hip_height * 0.98 if self.start_hip_height is not None else False
            is_stopped = abs(hip_velocity) < 0.1
            if is_at_top and is_stopped:
                rep_duration = current_time - self.current_rep_start_time if self.current_rep_start_time is not None else 0
                self.reps_completed += 1
                self.rep_details.append({
                    'rep_number': self.reps_completed, 'start_frame': self.current_rep_start_frame,
                    'end_frame': frame_idx, 'duration': rep_duration
                })
                self.rep_state = "standing"

    def draw_landmarks(self, frame, landmarks, state: str):
        color = (128, 128, 128) # Gray for 'searching'
        if state == 'analyzing':
            color = (0, 255, 0) # Green for 'analyzing'
        elif state == 'finished':
            color = (0, 0, 255) # Red for 'finished'

        selected_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        h, w = frame.shape[:2]
        for start_idx, end_idx in selected_connections:
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            start_pixel = (int(start_point.x * w), int(start_point.y * h))
            end_pixel = (int(end_point.x * w), int(end_point.y * h))
            cv2.line(frame, start_pixel, end_pixel, color, 2)
            cv2.circle(frame, start_pixel, 4, color, -1)
            cv2.circle(frame, end_pixel, 4, color, -1)
            
    def get_final_analysis(self) -> Dict:
        if not self.frame_metrics:
            return {'status': 'failed', 'message': 'No exercise detected.'}

        total_time = self.frame_metrics[-1]['time'] - self.frame_metrics[0]['time']
        avg_intensity = np.mean(self.avg_acceleration) if self.avg_acceleration else 0
        
        analysis_results = {
            'status': 'success',
            'exercise_period': {
                'start_frame': self.exercise_start_frame,
                'end_frame': self.exercise_start_frame + len(self.frame_metrics),
                'duration_seconds': total_time
            },
            'rep_counting': {'completed_reps': self.reps_completed, 'rep_details': self.rep_details},
            'metrics': {
                'volume': float(self.total_volume), 'max_intensity': float(self.max_acceleration),
                'avg_intensity': float(avg_intensity)
            },
            'time_series': self.frame_metrics
        }
        return self.convert_numpy_types(analysis_results)

    def convert_numpy_types(self, obj):
        """Recursively convert NumPy types to native Python types for JSON compatibility."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_): # <--- THIS LINE WAS MISSING
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(i) for i in obj]
        return obj