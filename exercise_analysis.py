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
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def get_3d_point(self, landmark) -> np.ndarray:
        """Convert MediaPipe landmark to 3D numpy array"""
        return np.array([landmark.x, landmark.y, landmark.z])

    def normalize_pose(self, landmarks) -> Dict[str, np.ndarray]:
        """Normalize pose to be invariant to camera position"""
        # Get hip points
        left_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        
        # Calculate hip center as origin
        hip_center = (left_hip + right_hip) / 2
        
        # Dictionary to store normalized points
        normalized_points = {}
        
        # List of landmarks we want to normalize - focusing only on key points
        landmark_list = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        # Normalize each point by centering on hip
        for landmark in landmark_list:
            point = self.get_3d_point(landmarks.landmark[landmark])
            normalized = point - hip_center
            normalized_points[landmark] = normalized
            
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

    def calculate_velocity(self, current_pos: float, prev_pos: float, fps: float) -> float:
        """Calculate velocity between two positions"""
        return (current_pos - prev_pos) * fps

    def calculate_acceleration(self, current_vel: float, prev_vel: float, fps: float) -> float:
        """Calculate acceleration between two velocities"""
        return (current_vel - prev_vel) * fps

    def calculate_joint_angles(self, normalized_points: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate relevant joint angles from normalized pose"""
        angles = {}
        
        # Left knee angle
        angles['left_knee'] = self.calculate_3d_angle(
            normalized_points[self.mp_pose.PoseLandmark.LEFT_HIP],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_KNEE],
            normalized_points[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        )
        
        # Right knee angle
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
        
        return angles

class SquatAnalyzer(ExerciseAnalyzer):
    """Analyzer specific to squat exercises focusing on intensity, volume, and time under tension"""
    def __init__(self):
        super().__init__()
        
        # Configuration parameters
        self.STANDING_KNEE_THRESHOLD = 160
        self.BOTTOM_KNEE_THRESHOLD = 110
        
        # Tracking variables for metrics
        self.is_active = False  # Track if user is in active exercise state
        self.start_time = 0  # For tracking time under tension
        self.total_tension_time = 0  # Total time under tension
        
        # Movement tracking
        self.frame_metrics = []
        self.total_volume = 0  # Total work volume (vertical distance × weight)
        self.accumulated_volume_over_time = []  # Track volume accumulation over time
        self.user_weight = 0  # Will be set from metadata
        self.max_acceleration = 0  # For tracking intensity
        self.avg_acceleration = []  # For tracking average intensity
        self.concentric_phase = False  # Track if in concentric (upward) phase
        
        # Previous state variables
        self.prev_knee_angle = None
        self.prev_velocity = None
        self.prev_hip_height = None
        self.prev_hip_velocity = 0
        
    def reset_analysis(self):
        """Reset analysis state for new video"""
        self.is_active = False
        self.start_time = 0
        self.total_tension_time = 0
        self.frame_metrics = []
        self.total_volume = 0
        self.accumulated_volume_over_time = []
        self.max_acceleration = 0
        self.avg_acceleration = []
        self.prev_knee_angle = None
        self.prev_velocity = None
        self.prev_hip_height = None
        self.prev_hip_velocity = 0
        self.concentric_phase = False
        
    def set_user_weight(self, load_kg: float):
        """Set the user's loaded weight for volume calculations"""
        self.user_weight = load_kg
        
    def detect_activity_state(self, knee_angle: float) -> bool:
        """Determine if the person is in an active exercise state based on knee angle"""
        # If knee angle is below the standing threshold, consider as active
        return knee_angle < self.STANDING_KNEE_THRESHOLD - 10  # 10 degree buffer
    
    def detect_movement_phase(self, hip_height: float) -> bool:
        """Determine if in concentric (upward) phase based on hip movement
        Returns: True for concentric (upward), False for eccentric (downward)
        """
        if self.prev_hip_height is None:
            return False
            
        # Moving upward = concentric phase
        return hip_height > self.prev_hip_height
            
    def analyze_frame(self, landmarks, frame_idx: int, fps: float, frame) -> Dict:
        """Analyze a single frame using normalized 3D pose"""
        if not landmarks:
            return None
            
        # Get normalized pose points
        normalized_points = self.normalize_pose(landmarks)
        
        # Calculate joint angles from normalized pose
        angles = self.calculate_joint_angles(normalized_points)
        
        # Calculate averages
        avg_knee_angle = (angles['left_knee'] + angles['right_knee']) / 2
        avg_hip_angle = (angles['left_hip'] + angles['right_hip']) / 2
        
        # Calculate hip center position (average of left and right hip)
        left_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center = (left_hip + right_hip) / 2
        hip_height = hip_center[1]  # Y-coordinate (vertical position)
        
        # Detect if concentric (upward) or eccentric (downward) phase
        is_concentric = self.detect_movement_phase(hip_height)
        
        # Calculate hip velocity and acceleration
        hip_velocity = 0
        hip_acceleration = 0
        intensity_value = 0  # Initialize with default value to avoid reference error
        
        if self.prev_hip_height is not None:
            # Calculate velocity in pixels per second (negative = moving up in image coordinates)
            hip_velocity = (hip_height - self.prev_hip_height) * fps
            
            # Calculate acceleration
            if self.prev_hip_velocity is not None:
                hip_acceleration = (hip_velocity - self.prev_hip_velocity) * fps
                
                # Intensity metric considers both phases but with different interpretations
                # - For concentric (upward): Higher acceleration is better
                # - For eccentric (downward): Lower acceleration is better (controlled)
                if is_concentric:
                    # For upward movement, we want high acceleration (more explosive)
                    intensity_value = abs(hip_acceleration)
                else:
                    # For downward movement, we want controlled descent (low acceleration)
                    # Invert the acceleration value so lower acceleration = higher score
                    intensity_value = 1.0 / (1.0 + abs(hip_acceleration))
                
                self.avg_acceleration.append(intensity_value)
                self.max_acceleration = max(self.max_acceleration, intensity_value)
        
        # Detect if in active state
        is_currently_active = self.detect_activity_state(avg_knee_angle)
        
        # Calculate time under tension
        if is_currently_active and not self.is_active:
            # Just entered active state
            self.is_active = True
            self.start_time = frame_idx / fps
        elif not is_currently_active and self.is_active:
            # Just exited active state
            self.is_active = False
            self.total_tension_time += (frame_idx / fps - self.start_time)
        
        # Calculate volume only in concentric (upward) phase
        if is_concentric and self.prev_hip_height is not None and self.user_weight > 0:
            # Calculate vertical distance moved in this frame
            vertical_distance = abs(self.prev_hip_height - hip_height)
            
            # Volume = vertical distance × weight (only count upward movement)
            frame_volume = vertical_distance * self.user_weight
            self.total_volume += frame_volume
        else:
            frame_volume = 0
            
        # Track accumulated volume over time
        current_accumulated_volume = self.total_volume
        self.accumulated_volume_over_time.append({
            'time': frame_idx / fps,
            'volume': current_accumulated_volume
        })
        
        # Store frame data
        frame_data = {
            'frame_idx': frame_idx,
            'avg_knee_angle': avg_knee_angle,
            'avg_hip_angle': avg_hip_angle,
            'hip_height': hip_height,
            'hip_velocity': hip_velocity,
            'hip_acceleration': hip_acceleration,
            'is_concentric': is_concentric,
            'phase_intensity': intensity_value,  # Now intensity_value is always defined
            'frame_volume': frame_volume,
            'accumulated_volume': current_accumulated_volume,
            'is_active': is_currently_active,
            'time': frame_idx / fps
        }
        
        self.frame_metrics.append(frame_data)
        self.prev_knee_angle = avg_knee_angle
        self.prev_hip_height = hip_height
        self.prev_hip_velocity = hip_velocity
        self.concentric_phase = is_concentric
        
        # Draw only specified landmarks on frame - no text
        self.draw_landmarks_only(frame, landmarks)
        
        return frame_data

    def draw_landmarks_only(self, frame, landmarks):
        """Draw only selected landmarks on the frame without text overlays"""
        # List of landmarks we want to show
        selected_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        # Draw connections manually with selected points only
        h, w = frame.shape[:2]
        for connection in selected_connections:
            start_idx, end_idx = connection
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            # Convert normalized coordinates to pixel coordinates
            start_pixel = (int(start_point.x * w), int(start_point.y * h))
            end_pixel = (int(end_point.x * w), int(end_point.y * h))
            
            # Draw line and circles at the connection points
            cv2.line(frame, start_pixel, end_pixel, (255, 0, 0), 1)
            cv2.circle(frame, start_pixel, 3, (0, 0, 0), -1)
            cv2.circle(frame, end_pixel, 3, (255, 255, 255), -1)
            
    def get_final_analysis(self) -> Dict:
        """Get the final analysis with focus on volume, intensity, and time under tension"""
        # If still in active state at the end, add the remaining time
        if self.is_active and self.frame_metrics:
            last_frame = self.frame_metrics[-1]
            self.total_tension_time += (last_frame['time'] - self.start_time)
        
        # Calculate average intensity
        avg_intensity = np.mean(self.avg_acceleration) if self.avg_acceleration else 0
        
        # Create time series data for graphs
        time_series = []
        
        # First, ensure frame_metrics are sorted by time
        self.frame_metrics = sorted(self.frame_metrics, key=lambda x: x['time'])
        
        # Create time series data from frame metrics
        for i, frame in enumerate(self.frame_metrics):
            # Convert NumPy types to native Python types
            time_series.append({
                'time': float(frame['time']),
                'angle': float(frame['avg_knee_angle']),
                'hip_height': float(frame['hip_height']),
                'hip_velocity': float(frame['hip_velocity']),
                'hip_acceleration': float(frame['hip_acceleration']),
                'phase_intensity': float(frame['phase_intensity']),
                'is_concentric': bool(frame['is_concentric']),
                'accumulated_volume': float(frame['accumulated_volume'])
            })
        
        # Determine time windows of tension for visualization
        in_tension = False
        tension_start = 0
        tension_windows = []
        
        # Track tension windows
        for i, frame in enumerate(self.frame_metrics):
            # Convert the is_active status to a simple bool to avoid any issues
            is_active = bool(frame['is_active'])
            
            # Track time under tension windows
            if is_active and not in_tension:
                # Just entered active state
                in_tension = True
                tension_start = float(frame['time'])
            elif not is_active and in_tension:
                # Just exited active state
                in_tension = False
                # Ensure we're using native float values
                tension_windows.append({
                    'start': float(tension_start),
                    'end': float(frame['time'])
                })
        
        # Close any open tension window
        if in_tension and self.frame_metrics:
            tension_windows.append({
                'start': float(tension_start),
                # Make sure we use the last frame time
                'end': float(self.frame_metrics[-1]['time'])
            })
        
        # Validate and clean tension windows
        cleaned_tension_windows = []
        for window in tension_windows:
            # Skip invalid windows
            if window['start'] >= window['end']:
                continue
            # Ensure no overlap with existing windows
            overlapping = False
            for existing in cleaned_tension_windows:
                if (window['start'] <= existing['end'] and 
                    window['end'] >= existing['start']):
                    overlapping = True
                    break
            if not overlapping:
                cleaned_tension_windows.append(window)
        
        # Assign the cleaned windows
        tension_windows = cleaned_tension_windows
        
        # Log the tension windows for debugging
        logging.info(f"Calculated tension windows: {tension_windows}")
        
        # Get volume over time
        volume_over_time = self.accumulated_volume_over_time
        
        # Convert volume_over_time to native Python types
        converted_volume_progression = []
        for point in volume_over_time:
            converted_volume_progression.append({
                'time': float(point['time']),
                'volume': float(point['volume'])
            })
        
        # Calculate total score based on research benchmarks for muscle hypertrophy
        # Reference benchmarks (based on literature for muscle hypertrophy)
        # - Ideal time under tension: 40-70 seconds per set
        # - Optimal volume: Varies based on exercise, but roughly 10-20 kg·m for squats
        # - Ideal intensity: Varies, using normalized score
        
        # Score calculations (0-100 scale for each component)
        volume_score = min(100, self.total_volume / 15.0 * 100)  # Normalized to 15 kg·m as target
        
        # Time under tension score (bell curve with optimal range of 40-70 seconds)
        tut_score = 0
        if self.total_tension_time < 20:
            tut_score = max(0, self.total_tension_time / 20.0 * 50)  # 0-50 for 0-20 seconds
        elif 20 <= self.total_tension_time < 40:
            tut_score = 50 + (self.total_tension_time - 20) / 20.0 * 50  # 50-100 for 20-40 seconds
        elif 40 <= self.total_tension_time <= 70:
            tut_score = 100  # Optimal range gets full score
        elif 70 < self.total_tension_time <= 120:
            tut_score = 100 - (self.total_tension_time - 70) / 50.0 * 30  # 100-70 for 70-120 seconds
        else:
            tut_score = max(0, 70 - (self.total_tension_time - 120) / 60.0 * 70)  # 70-0 for 120+ seconds
        
        # Intensity score (normalized to benchmark)
        intensity_score = min(100, avg_intensity * 50)  # Scale based on typical values
        
        # Weighted total score - give slightly more weight to volume and TUT
        total_score = (volume_score * 0.4) + (intensity_score * 0.25) + (tut_score * 0.35)
        
        # Prepare the final analysis with updated metrics
        analysis_results = {
            'status': 'success',
            'metrics': {
                'volume': float(self.total_volume),
                'volume_unit': 'kg·m',  # kilogram-meters
                'max_intensity': float(self.max_acceleration),
                'avg_intensity': float(avg_intensity),
                'time_under_tension': float(self.total_tension_time),
                'volume_score': float(volume_score),
                'intensity_score': float(intensity_score),
                'tut_score': float(tut_score),
                'total_score': float(total_score)
            },
            'time_series': time_series,
            'volume_progression': converted_volume_progression,
            'tension_windows': tension_windows
        }
        
        # Convert any remaining NumPy types
        return convert_numpy_types(analysis_results)

# Utility function outside of the class
def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for Firestore compatibility"""
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