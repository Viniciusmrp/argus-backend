from abc import ABC, abstractmethod
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List

class BaseAnalyzer(ABC):
    """
    Abstract base class for exercise analyzers.
    This class provides a generic framework for processing pose data,
    calculating kinematic metrics, and analyzing repetitions.
    """

    def __init__(self):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Kinematic and metric tracking
        self.prev_world_landmarks = None
        self.prev_world_velocities = None
        self.frame_metrics = []
        self.total_volume = 0
        self.user_weight = 0

    @abstractmethod
    def get_body_landmarks(self) -> List[int]:
        """Return the list of MediaPipe PoseLandmark indices relevant to this exercise."""
        pass

    def get_3d_point(self, landmark) -> np.ndarray:
        """Convert a MediaPipe landmark to a 3D numpy array."""
        return np.array([landmark.x, landmark.y, landmark.z])

    def get_body_coordinate_system(self, landmarks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates a coordinate system based on the person's body orientation.
        Returns three orthogonal vectors representing the new Y, Z, and X axes.
        """
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

    def normalize_pose(self, landmarks) -> Dict[int, np.ndarray]:
        """Normalizes pose to be invariant to camera position and rotation."""
        left_hip_3d = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_3d = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center = (left_hip_3d + right_hip_3d) / 2

        y_axis, z_axis, x_axis = self.get_body_coordinate_system(landmarks)
        normalized_points = {}

        for landmark_idx in self.get_body_landmarks():
            point_3d = self.get_3d_point(landmarks.landmark[landmark_idx])
            centered_point = point_3d - hip_center
            
            new_x = np.dot(centered_point, x_axis)
            new_y = np.dot(centered_point, y_axis)
            new_z = np.dot(centered_point, z_axis)

            normalized_points[landmark_idx] = np.array([new_x, new_y, new_z])
        return normalized_points

    def calculate_3d_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate the angle between three 3D points."""
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_velocity(self, current_pos: np.ndarray, prev_pos: np.ndarray, fps: float) -> np.ndarray:
        """Calculate velocity between two 3D positions."""
        return (current_pos - prev_pos) * fps

    def calculate_acceleration(self, current_vel: np.ndarray, prev_vel: np.ndarray, fps: float) -> np.ndarray:
        """Calculate acceleration between two 3D velocities."""
        return (current_vel - prev_vel) * fps

    @abstractmethod
    def calculate_joint_angles(self, normalized_points: Dict[int, np.ndarray]) -> Dict[str, float]:
        """Calculate relevant joint angles for the specific exercise."""
        pass

    def calculate_kinematics(self, world_landmarks, fps: float) -> Tuple[Dict, Dict]:
        """Calculate velocities and accelerations for all relevant body landmarks."""
        world_velocities = {}
        world_accelerations = {}
        body_landmarks = self.get_body_landmarks()

        if self.prev_world_landmarks:
            for landmark_idx in body_landmarks:
                current_pos = self.get_3d_point(world_landmarks.landmark[landmark_idx])
                prev_pos = self.get_3d_point(self.prev_world_landmarks.landmark[landmark_idx])
                velocity = self.calculate_velocity(current_pos, prev_pos, fps)
                world_velocities[landmark_idx.name] = velocity
                
                if self.prev_world_velocities:
                    prev_velocity = self.prev_world_velocities.get(landmark_idx.name)
                    if prev_velocity is not None:
                        acceleration = self.calculate_acceleration(velocity, prev_velocity, fps)
                        world_accelerations[landmark_idx.name] = acceleration
                        
        return world_velocities, world_accelerations

    def analyze_frame(self, landmarks, world_landmarks, frame_idx: int, fps: float, frame):
        """Generic frame analysis to be called by subclasses."""
        if not landmarks or not world_landmarks:
            return None

        # Kinematic calculations
        world_velocities, world_accelerations = self.calculate_kinematics(world_landmarks, fps)
        
        # Pose and angle calculations
        normalized_points = self.normalize_pose(landmarks)
        angles = self.calculate_joint_angles(normalized_points)
        
        # Exercise-specific processing
        frame_data = self.process_exercise_specific_metrics(
            frame_idx, fps, normalized_points, angles, world_velocities, world_accelerations, world_landmarks
        )
        
        # Update state for next frame
        self.prev_world_landmarks = world_landmarks
        self.prev_world_velocities = world_velocities

        # Visualization
        self.draw_landmarks(frame, landmarks, frame_data)
        
        # Store frame metrics
        self.frame_metrics.append(frame_data)
        
        return frame_data

    @abstractmethod
    def process_exercise_specific_metrics(self, frame_idx: int, fps: float, normalized_points: Dict, angles: Dict, 
                                          world_velocities: Dict, world_accelerations: Dict, world_landmarks) -> Dict:
        """Process metrics unique to the specific exercise (e.g., rep counting)."""
        pass
    
    @abstractmethod
    def draw_landmarks(self, frame, landmarks, frame_data: Dict):
        """Draw landmarks and exercise-specific feedback on the frame."""
        pass

    @abstractmethod
    def get_final_analysis(self) -> Dict:
        """Return the final analysis summary."""
        pass

    def reset_analysis(self):
        """Reset analysis state for a new video."""
        self.prev_world_landmarks = None
        self.prev_world_velocities = None
        self.frame_metrics = []
        self.total_volume = 0

    def set_user_weight(self, load_kg: float):
        """Set the user's loaded weight for volume calculations."""
        self.user_weight = load_kg

    def convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON compatibility."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
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
