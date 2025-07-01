import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
import logging
from collections import deque

class ExerciseAnalyzer:
    """Base class for exercise analysis with 3D pose normalization"""
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
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
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
    """Analyzer specific to squat exercises with improved exercise detection and rep counting"""
    def __init__(self):
        super().__init__()
        
        # Enhanced thresholds for better detection
        self.STANDING_KNEE_THRESHOLD = 150  # Minimum angle to be considered standing
        self.BOTTOM_KNEE_THRESHOLD = 100    # Maximum angle to be considered in bottom position
        self.MIN_DEPTH_THRESHOLD = 110      # Minimum depth required for a valid rep
        
        # Exercise state detection parameters
        self.EXERCISE_DETECTION_WINDOW = 30  # frames to confirm exercise state
        self.MIN_CONSECUTIVE_ACTIVE_FRAMES = 15  # minimum frames to confirm exercise start
        self.MIN_CONSECUTIVE_INACTIVE_FRAMES = 60  # minimum frames to confirm exercise end
        
        # Rep counting parameters
        self.REP_CONFIRMATION_FRAMES = 10   # frames to confirm a rep transition
        self.MIN_REP_DURATION = 1.0         # minimum seconds for a valid rep
        self.MAX_REP_DURATION = 8.0         # maximum seconds for a valid rep
        
        # State tracking with sliding windows
        self.activity_window = deque(maxlen=self.EXERCISE_DETECTION_WINDOW)
        self.knee_angle_history = deque(maxlen=60)  # 2 seconds at 30fps
        self.hip_height_history = deque(maxlen=60)
        
        # Exercise state tracking
        self.exercise_state = "inactive"  # "inactive", "starting", "active", "ending"
        self.consecutive_active_frames = 0
        self.consecutive_inactive_frames = 0
        self.exercise_start_frame = None
        self.exercise_end_frame = None
        
        # Rep counting state machine
        self.rep_state = "standing"  # "standing", "descending", "ascending"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.start_hip_height = None
        self.reps_completed = 0
        self.rep_details = []  # Store details about each rep
        
        # Tracking variables for metrics (only during active exercise)
        self.is_analyzing = False  # Only analyze when exercise is confirmed active
        self.analysis_start_frame = None
        self.start_time = 0
        self.total_tension_time = 0
        
        # Movement tracking (only during analysis)
        self.frame_metrics = []
        self.total_volume = 0
        self.accumulated_volume_over_time = []
        self.user_weight = 0
        self.max_acceleration = 0
        self.avg_acceleration = []
        self.concentric_phase = False
        self.standing_confirmation_frames = 0
        
        # Previous state variables
        self.prev_knee_angle = None
        self.prev_velocity = None
        self.prev_hip_height = None
        self.prev_hip_velocity = 0
        
    def reset_analysis(self):
        """Reset analysis state for new video"""
        # Clear all tracking variables
        self.activity_window.clear()
        self.knee_angle_history.clear()
        self.hip_height_history.clear()
        
        # Reset exercise state
        self.exercise_state = "inactive"
        self.consecutive_active_frames = 0
        self.consecutive_inactive_frames = 0
        self.exercise_start_frame = None
        self.exercise_end_frame = None
        
        # Reset rep counting
        self.rep_state = "standing"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.start_hip_height = None
        self.reps_completed = 0
        self.rep_details = []
        
        # Reset analysis tracking
        self.is_analyzing = False
        self.analysis_start_frame = None
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
        
    def detect_potential_squat_movement(self, knee_angle: float, hip_height: float) -> bool:
        """
        Detect if current pose indicates potential squat movement
        More sophisticated than just knee angle
        """
        # Add to history
        self.knee_angle_history.append(knee_angle)
        self.hip_height_history.append(hip_height)
        
        # Need some history to make decisions
        if len(self.knee_angle_history) < 10:
            return False
            
        # Check if there's meaningful movement variation
        knee_variation = np.std(list(self.knee_angle_history)[-10:])
        hip_variation = np.std(list(self.hip_height_history)[-10:])
        
        # Check if person is in a reasonable squat-like position
        is_squat_position = (80 <= knee_angle <= 175)  # Broader range for detection
        has_movement = knee_variation > 2 or hip_variation > 0.005  # Some movement detected
        
        # Check if knee angle suggests squat-like movement (not just standing still)
        recent_knee_angles = list(self.knee_angle_history)[-5:]
        min_recent_knee = min(recent_knee_angles)
        max_recent_knee = max(recent_knee_angles)
        has_knee_flexion = (max_recent_knee - min_recent_knee) > 5  # At least 5 degrees of movement
        
        return is_squat_position and has_movement and has_knee_flexion
    
    def update_exercise_state(self, knee_angle: float, hip_height: float, frame_idx: int) -> str:
        """
        Update exercise state based on movement patterns
        Returns current exercise state
        """
        is_potential_exercise = self.detect_potential_squat_movement(knee_angle, hip_height)
        self.activity_window.append(is_potential_exercise)
        
        # Calculate activity ratio in recent window
        if len(self.activity_window) == self.EXERCISE_DETECTION_WINDOW:
            activity_ratio = sum(self.activity_window) / len(self.activity_window)
        else:
            activity_ratio = 0
            
        # State machine for exercise detection
        if self.exercise_state == "inactive":
            if activity_ratio > 0.3:  # 30% of recent frames show exercise-like movement
                self.consecutive_active_frames += 1
                if self.consecutive_active_frames >= self.MIN_CONSECUTIVE_ACTIVE_FRAMES:
                    self.exercise_state = "starting"
                    self.exercise_start_frame = frame_idx
                    logging.info(f"Exercise starting detected at frame {frame_idx}")
            else:
                self.consecutive_active_frames = 0
                
        elif self.exercise_state == "starting":
            if activity_ratio > 0.7:  # Higher threshold to confirm
                self.exercise_state = "active"
                self.analysis_start_frame = frame_idx
                self.is_analyzing = True
                logging.info(f"Exercise confirmed active at frame {frame_idx}")
            elif activity_ratio < 0.3:  # False start
                self.exercise_state = "inactive"
                self.consecutive_active_frames = 0
                
        elif self.exercise_state == "active":
            if activity_ratio < 0.3:  # Low activity suggests exercise ending
                self.consecutive_inactive_frames += 1
                if self.consecutive_inactive_frames >= self.MIN_CONSECUTIVE_INACTIVE_FRAMES:
                    self.exercise_state = "ending"
            else:
                self.consecutive_inactive_frames = 0
                
        elif self.exercise_state == "ending":
            if activity_ratio > 0.5:  # Activity resumed
                self.exercise_state = "active"
                self.consecutive_inactive_frames = 0
            else:
                # Confirm exercise has ended
                self.exercise_state = "inactive"
                self.exercise_end_frame = frame_idx
                self.is_analyzing = False
                logging.info(f"Exercise ended at frame {frame_idx}")
                
        return self.exercise_state
    
    def count_reps(self, hip_height: float, hip_velocity: float, frame_idx: int, fps: float) -> Dict:
        """
        More flexible rep counting based on hip movement dynamics.
        """
        rep_info = {
            'rep_completed': False,
            'rep_state': self.rep_state,
            'current_reps': self.reps_completed
        }

        if not self.is_analyzing:
            return rep_info

        current_time = frame_idx / fps

        if self.rep_state == "standing":
            # Check for start of descent (consistent downward hip movement)
            if hip_velocity < -0.05:
                self.rep_state = "descending"
                self.current_rep_start_frame = frame_idx
                self.current_rep_start_time = current_time
                self.start_hip_height = hip_height
                self.standing_confirmation_frames = 0
                logging.info(f"Rep descent started at frame {frame_idx}")

        elif self.rep_state == "descending":
            # Check for bottom of squat (hip velocity changes from negative to positive)
            if hip_velocity >= 0:
                self.rep_state = "ascending"
                logging.info(f"Bottom position reached at frame {frame_idx}")

        elif self.rep_state == "ascending":
            # Check if user has returned to the top and stopped moving
            is_at_top = hip_height >= self.start_hip_height * 0.98
            is_stopped = abs(hip_velocity) < 0.1

            if is_at_top and is_stopped:
                self.standing_confirmation_frames += 1
            else:
                self.standing_confirmation_frames = 0

            if self.standing_confirmation_frames >= self.REP_CONFIRMATION_FRAMES:
                rep_duration = current_time - self.current_rep_start_time

                if self.MIN_REP_DURATION <= rep_duration <= self.MAX_REP_DURATION:
                    self.reps_completed += 1
                    rep_info['rep_completed'] = True
                    
                    rep_detail = {
                        'rep_number': self.reps_completed,
                        'start_frame': self.current_rep_start_frame,
                        'end_frame': frame_idx,
                        'duration': rep_duration,
                        'start_time': self.current_rep_start_time,
                        'end_time': current_time
                    }
                    self.rep_details.append(rep_detail)
                    logging.info(f"Rep {self.reps_completed} completed in {rep_duration:.2f}s")
                
                self.rep_state = "standing"
                self.standing_confirmation_frames = 0

            # Handle reversal of direction during ascent
            elif hip_velocity < -0.05:
                logging.info("Reversed direction during ascent.")
                self.rep_state = "descending"
                self.current_rep_start_frame = frame_idx
                self.current_rep_start_time = current_time
                self.start_hip_height = hip_height
                self.standing_confirmation_frames = 0


        rep_info.update({
            'rep_state': self.rep_state,
            'current_reps': self.reps_completed
        })

        return rep_info
            
    def detect_movement_phase(self, hip_height: float) -> bool:
        """Determine if in concentric (upward) phase based on hip movement"""
        if self.prev_hip_height is None:
            return False
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
        
        # Calculate hip center position
        left_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = self.get_3d_point(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center = (left_hip + right_hip) / 2
        hip_height = hip_center[1]

        # Calculate hip velocity
        hip_velocity = 0
        if self.prev_hip_height is not None:
            hip_velocity = (hip_height - self.prev_hip_height) * fps
        
        # Update exercise state
        exercise_state = self.update_exercise_state(avg_knee_angle, hip_height, frame_idx)
        
        # Count reps (only during active exercise)
        rep_info = self.count_reps(hip_height, hip_velocity, frame_idx, fps)
        
        # Only perform detailed analysis when exercise is active
        intensity_value = 0
        hip_acceleration = 0
        frame_volume = 0
        
        if self.is_analyzing:
            # Detect movement phase
            is_concentric = self.detect_movement_phase(hip_height)
            
            # Calculate hip acceleration
            if self.prev_hip_velocity is not None:
                hip_acceleration = (hip_velocity - self.prev_hip_velocity) * fps
                
                if is_concentric:
                    intensity_value = abs(hip_acceleration)
                else:
                    intensity_value = 1.0 / (1.0 + abs(hip_acceleration))
                
                self.avg_acceleration.append(intensity_value)
                self.max_acceleration = max(self.max_acceleration, intensity_value)
            
            # Calculate volume only in concentric phase during active exercise
            if is_concentric and self.prev_hip_height is not None and self.user_weight > 0:
                vertical_distance = abs(self.prev_hip_height - hip_height)
                frame_volume = vertical_distance * self.user_weight
                self.total_volume += frame_volume
            
            # Track accumulated volume over time
            self.accumulated_volume_over_time.append({
                'time': frame_idx / fps,
                'volume': self.total_volume
            })
            
            # Store frame data only during analysis
            frame_data = {
                'frame_idx': frame_idx,
                'avg_knee_angle': avg_knee_angle,
                'avg_hip_angle': avg_hip_angle,
                'hip_height': hip_height,
                'hip_velocity': hip_velocity,
                'hip_acceleration': hip_acceleration,
                'is_concentric': is_concentric,
                'phase_intensity': intensity_value,
                'frame_volume': frame_volume,
                'accumulated_volume': self.total_volume,
                'is_analyzing': self.is_analyzing,
                'exercise_state': exercise_state,
                'rep_state': rep_info['rep_state'],
                'current_reps': rep_info['current_reps'],
                'time': frame_idx / fps
            }
            
            self.frame_metrics.append(frame_data)
            self.concentric_phase = is_concentric
        
        # Always update previous values
        self.prev_knee_angle = avg_knee_angle
        self.prev_hip_height = hip_height
        self.prev_hip_velocity = hip_velocity
        
        # Draw landmarks based on exercise state
        self.draw_landmarks_with_state(frame, landmarks, exercise_state, rep_info)
        
        return {
            'exercise_state': exercise_state,
            'is_analyzing': self.is_analyzing,
            'rep_info': rep_info,
            'avg_knee_angle': avg_knee_angle
        }

    def draw_landmarks_with_state(self, frame, landmarks, exercise_state: str, rep_info: Dict):
        """Draw landmarks with different colors based on exercise state"""
        # Color coding based on state
        if exercise_state == "inactive":
            line_color = (128, 128, 128)  # Gray
            point_color = (64, 64, 64)    # Dark gray
        elif exercise_state in ["starting", "ending"]:
            line_color = (0, 255, 255)    # Yellow
            point_color = (0, 128, 255)   # Orange
        else:  # active
            line_color = (0, 255, 0)      # Green
            point_color = (255, 0, 0)     # Blue
        
        # List of connections to draw
        selected_connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        # Draw connections
        h, w = frame.shape[:2]
        for connection in selected_connections:
            start_idx, end_idx = connection
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            start_pixel = (int(start_point.x * w), int(start_point.y * h))
            end_pixel = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start_pixel, end_pixel, line_color, 2)
            cv2.circle(frame, start_pixel, 4, point_color, -1)
            cv2.circle(frame, end_pixel, 4, point_color, -1)
            
    def get_final_analysis(self) -> Dict:
        """Get the final analysis with enhanced rep counting and exercise detection"""
        # Calculate metrics only from active exercise periods
        if self.is_analyzing and self.frame_metrics:
            last_frame = self.frame_metrics[-1]
            if self.analysis_start_frame:
                self.total_tension_time = last_frame['time'] - (self.analysis_start_frame / 30.0)  # Approximate fps
        
        # Calculate average intensity from active periods only
        avg_intensity = np.mean(self.avg_acceleration) if self.avg_acceleration else 0
        
        # Create time series data only from analysis periods
        time_series = []
        for frame in sorted(self.frame_metrics, key=lambda x: x['time']):
            time_series.append({
                'time': float(frame['time']),
                'angle': float(frame['avg_knee_angle']),
                'hip_height': float(frame['hip_height']),
                'hip_velocity': float(frame['hip_velocity']),
                'hip_acceleration': float(frame['hip_acceleration']),
                'phase_intensity': float(frame['phase_intensity']),
                'is_concentric': bool(frame['is_concentric']),
                'accumulated_volume': float(frame['accumulated_volume']),
                'exercise_state': frame['exercise_state'],
                'rep_state': frame['rep_state'],
                'current_reps': int(frame['current_reps'])
            })
        
        # Calculate volume over time
        volume_over_time = []
        for point in self.accumulated_volume_over_time:
            volume_over_time.append({
                'time': float(point['time']),
                'volume': float(point['volume'])
            })
        
        # Enhanced scoring with rep accuracy bonus
        volume_score = min(100, self.total_volume / 15.0 * 100) if self.total_volume > 0 else 0
        
        # TUT score calculation
        tut_score = 0
        if self.total_tension_time > 0:
            if self.total_tension_time < 20:
                tut_score = max(0, self.total_tension_time / 20.0 * 50)
            elif 20 <= self.total_tension_time < 40:
                tut_score = 50 + (self.total_tension_time - 20) / 20.0 * 50
            elif 40 <= self.total_tension_time <= 70:
                tut_score = 100
            elif 70 < self.total_tension_time <= 120:
                tut_score = 100 - (self.total_tension_time - 70) / 50.0 * 30
            else:
                tut_score = max(0, 70 - (self.total_tension_time - 120) / 60.0 * 70)
        
        intensity_score = min(100, avg_intensity * 50) if avg_intensity > 0 else 0
        
        # Weighted total score
        total_score = (volume_score * 0.4) + (intensity_score * 0.3) + (tut_score * 0.3)
        
        analysis_results = {
            'status': 'success',
            'exercise_detection': {
                'exercise_start_frame': self.exercise_start_frame,
                'exercise_end_frame': self.exercise_end_frame,
                'analysis_start_frame': self.analysis_start_frame,
                'total_frames_analyzed': len(self.frame_metrics)
            },
            'rep_counting': {
                'completed_reps': self.reps_completed,
                'rep_details': self.rep_details
            },
            'metrics': {
                'volume': float(self.total_volume),
                'volume_unit': 'kg·m',
                'max_intensity': float(self.max_acceleration),
                'avg_intensity': float(avg_intensity),
                'time_under_tension': float(self.total_tension_time),
                'volume_score': float(volume_score),
                'intensity_score': float(intensity_score),
                'tut_score': float(tut_score),
                'total_score': float(total_score)
            },
            'time_series': time_series,
            'volume_progression': volume_over_time
        }
        
        return self.convert_numpy_types(analysis_results)

    def convert_numpy_types(self, obj):
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
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj