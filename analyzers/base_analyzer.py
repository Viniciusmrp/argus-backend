from abc import ABC, abstractmethod
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple

class BaseAnalyzer(ABC):
    """
    Abstract base class for exercise analyzers.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def get_3d_point(self, landmark) -> np.ndarray:
        """Convert MediaPipe landmark to 3D numpy array"""
        return np.array([landmark.x, landmark.y, landmark.z])

    def get_body_coordinate_system(self, landmarks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates a coordinate system based on the person's body orientation. Returns three orthogonal vectors representing the new Y, Z, and X axes."""
        left_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        left_shoulder = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])

        # Y-axis (Up): Vector from hip center to shoulder center
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        y_axis = shoulder_center - hip_center
        y_axis /= np.linalg.norm(y_axis)

        # Z-axis (Forward/Backward): Vector perpendicular to the hips and the body's up-direction.
        # We use the cross product to find a vector orthogonal to both the hip line and the up vector.
        hip_line = right_hip - left_hip
        z_axis = np.cross(y_axis, hip_line)
        z_axis /= np.linalg.norm(z_axis)

        # X-axis (Right/Left): Vector perpendicular to the Y and Z axes
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        return y_axis, z_axis, x_axis

    def normalize_pose(self, landmarks) -> Dict[str, np.ndarray]:
        """Normalize pose to be invariant to camera position and rotation."""
        left_hip_3d = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_3d = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center = (left_hip_3d + right_hip_3d) / 2

        # 1. Get the body's coordinate system axes for the current frame
        y_axis, z_axis, x_axis = self.get_body_coordinate_system(landmarks)

        # Dictionary to store the newly calculated points
        normalized_points = {}

        # List of landmarks we want to transform into the new coordinate system
        landmark_list = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.LEFT_THUMB,
            self.mp_pose.PoseLandmark.RIGHT_THUMB
        ]

        for landmark_idx in landmark_list:
            point_3d = self.get_3d_point(landmarks.landmark[landmark_idx])

            # 2. Center the point around the hip origin
            centered_point = point_3d - hip_center

            # 3. Project the centered point onto the new axes to get the body-centric coordinates
            new_x = np.dot(centered_point, x_axis)
            new_y = np.dot(centered_point, y_axis)
            new_z = np.dot(centered_point, z_axis)

            normalized_points[landmark_idx] = np.array([new_x, new_y, new_z])

        return normalized_points

    def calculate_3d_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three 3D points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

    def calculate_velocity(self, current_pos: np.ndarray, prev_pos: np.ndarray, fps: float) -> np.ndarray:
        """Calculate velocity between two 3D positions"""
        if current_pos is None or prev_pos is None:
            return np.array([0, 0, 0])
        return (current_pos - prev_pos) * fps

    def calculate_acceleration(self, current_vel: np.ndarray, prev_vel: np.ndarray, fps: float) -> np.ndarray:
        """Calculate acceleration between two 3D velocities"""
        if current_vel is None or prev_vel is None:
            return np.array([0, 0, 0])
        return (current_vel - prev_vel) * fps

    def calculate_joint_angles(self, normalized_points: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate relevant joint angles from normalized pose"""
        angles = {}
        
        # Knee angles
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
        
        # Hip angles
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
        
        # Ankle angles
        angles['left_ankle'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_KNEE],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_ANKLE],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        )
        angles['right_ankle'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_KNEE],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_ANKLE],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        )
        
        # Shoulder angles
        angles['left_shoulder'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        )
        angles['right_shoulder'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        )
        
        # Elbow angles
        angles['left_elbow'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_ELBOW],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_WRIST]
        )
        angles['right_elbow'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        )
        
        # Wrist angles
        angles['left_wrist'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_ELBOW],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_WRIST],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_THUMB]
        )
        angles['right_wrist'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_WRIST],
            normalized_points[self.mp_pose.PoseLandmark.RIGHT_THUMB]
        )
        
        return angles

    @abstractmethod
    def analyze_frame(self, landmarks, world_landmarks, frame_idx: int, fps: float, frame):
        pass

    @abstractmethod
    def get_final_analysis(self) -> Dict:
        pass

    def reset_analysis(self):
        """Reset analysis state for new video"""
        pass
        
    def set_user_weight(self, load_kg: float):
        """Set the user's loaded weight for volume calculations"""
        pass
