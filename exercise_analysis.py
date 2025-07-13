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
        """Calculate angle between three 3D points."""
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

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
    """Analyzer specific to squat exercises, adapted for world landmarks."""
    def __init__(self):
        super().__init__()
        self.STANDING_KNEE_THRESHOLD = 150
        self.BOTTOM_KNEE_THRESHOLD = 100
        self.MIN_DEPTH_THRESHOLD = 110
        self.EXERCISE_DETECTION_WINDOW = 20
        self.MIN_CONSECUTIVE_ACTIVE_FRAMES = 5
        self.MIN_CONSECUTIVE_INACTIVE_FRAMES = 60
        self.REP_CONFIRMATION_FRAMES = 3
        self.MIN_REP_DURATION = 0.8
        self.MAX_REP_DURATION = 8.0
        self.activity_window = deque(maxlen=self.EXERCISE_DETECTION_WINDOW)
        self.knee_angle_history = deque(maxlen=60)
        self.hip_height_history = deque(maxlen=60)
        self.exercise_state = "inactive"
        self.consecutive_active_frames = 0
        self.consecutive_inactive_frames = 0
        self.exercise_start_frame = None
        self.exercise_end_frame = None
        self.rep_state = "standing"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.start_hip_height = None
        self.reps_completed = 0
        self.rep_details = []
        self.is_analyzing = False
        self.analysis_start_frame = None
        self.start_time = 0
        self.total_tension_time = 0
        self.frame_metrics = []
        self.total_volume = 0
        self.accumulated_volume_over_time = []
        self.user_weight = 0
        self.max_acceleration = 0
        self.avg_acceleration = []
        self.concentric_phase = False
        self.standing_confirmation_frames = 0
        self.prev_knee_angle = None
        self.prev_velocity = None
        self.prev_hip_height = None
        self.prev_hip_velocity = 0

    # **** THIS IS THE FIX ****
    # Add the missing method back into the class
    def set_user_weight(self, load_kg: float):
        """Set the user's loaded weight for volume calculations"""
        self.user_weight = load_kg

    def analyze_frame(self, pose_results, frame_idx: int, fps: float, frame) -> Optional[Dict]:
        if not pose_results.pose_world_landmarks or not pose_results.pose_landmarks:
            self.prev_world_landmarks = None
            self.prev_velocities = {}
            return None

        world_landmarks = pose_results.pose_world_landmarks
        image_landmarks = pose_results.pose_landmarks

        joint_angles = self.calculate_all_joint_angles(world_landmarks)
        velocities, accelerations = self.calculate_kinetics(world_landmarks, fps)

        # Your squat analysis logic will go here. For now, we are just returning the raw data.
        # This part will need further adaptation.

        self.prev_world_landmarks = world_landmarks
        self.prev_velocities = velocities

        # Draw landmarks using image_landmarks for correct overlay
        self.draw_landmarks_with_state(frame, image_landmarks, self.exercise_state, {'rep_state': self.rep_state, 'current_reps': self.reps_completed})

        return {
            'frame_index': frame_idx,
            'timestamp': frame_idx / fps,
            'joint_angles': joint_angles,
            'velocities': {name: vel.tolist() for name, vel in velocities.items()},
            'accelerations': {name: acc.tolist() for name, acc in accelerations.items()}
        }

    # You will need to adapt the logic inside these methods to use the new detailed data
    def draw_landmarks_with_state(self, frame, landmarks, exercise_state: str, rep_info: Dict):
        """Draw landmarks with different colors based on exercise state"""
        # (Your existing drawing logic here)
        line_color = (128, 128, 128)
        if exercise_state == "active":
            line_color = (0, 255, 0)
        
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
        for connection in selected_connections:
            start_idx, end_idx = connection
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            start_pixel = (int(start_point.x * w), int(start_point.y * h))
            end_pixel = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start_pixel, end_pixel, line_color, 2)

    def get_final_analysis(self) -> Dict:
        # This method also needs to be adapted to process the new `frame_metrics` format
        return {"status": "success", "metrics": self.frame_metrics}