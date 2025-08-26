from collections import deque
import logging
import cv2
import numpy as np
from scipy.signal import savgol_filter
from analyzers.base_analyzer import BaseAnalyzer
from typing import Dict

class SquatAnalyzer(BaseAnalyzer):
    """A simplified analyzer for squat exercises focusing directly on rep counting."""
    def __init__(self):
        super().__init__()
        
        # Rep counting parameters
        self.MIN_REP_DURATION = 0.8         # minimum seconds for a valid rep
        self.MAX_REP_DURATION = 8.0         # maximum seconds for a valid rep
        
        # State tracking
        self.knee_angle_history = deque(maxlen=10)
        self.hip_height_history = deque(maxlen=10)
        
        # Rep counting state machine
        self.rep_state = "standing"  # "standing", "descending", "ascending"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.reps_completed = 0
        self.rep_details = []
        self.calibrated_standing_knee_angle = None # Dynamically calibrated standing angle
        
        # Metrics tracking
        self.is_analyzing = True  # Always analyzing
        self.analysis_start_frame = 0
        self.start_time = 0
        self.total_tension_time = 0
        self.frame_metrics = []
        self.total_volume = 0
        self.accumulated_volume_over_time = []
        self.user_weight = 0
        self.max_acceleration = 0
        self.avg_acceleration = []
        self.concentric_phase = False
        
        # Previous state variables
        self.prev_world_landmarks = None
        self.prev_world_velocities = None
        self.prev_knee_angle = None
        self.prev_hip_height = None
        
    def reset_analysis(self):
        """Reset analysis state for a new video."""
        self.knee_angle_history.clear()
        self.hip_height_history.clear()
        
        # Reset rep counting
        self.rep_state = "standing"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.reps_completed = 0
        self.rep_details = []
        self.calibrated_standing_knee_angle = None
        
        # Reset analysis tracking
        self.is_analyzing = True
        self.analysis_start_frame = 0
        self.start_time = 0
        self.total_tension_time = 0
        self.frame_metrics = []
        self.total_volume = 0
        self.accumulated_volume_over_time = []
        self.max_acceleration = 0
        self.avg_acceleration = []
        self.prev_knee_angle = None
        self.prev_hip_height = None
        self.concentric_phase = False
        self.prev_world_landmarks = None
        self.prev_world_velocities = None
        
    def set_user_weight(self, load_kg: float):
        """Set the user's loaded weight for volume calculations."""
        self.user_weight = load_kg
        
    def get_knee_angle_trend(self) -> str:
        """Determines the trend of knee angle movement."""
        if len(self.knee_angle_history) < 5:
            return "stationary"

        x = np.arange(len(self.knee_angle_history))
        y = np.array(self.knee_angle_history)
        slope, _ = np.polyfit(x, y, 1)

        if abs(slope) < 0.5:
            return "stationary"
        elif slope > 0:
            return "ascending"
        else:
            return "descending"

    def count_reps(self, avg_knee_angle: float, frame_idx: int, fps: float) -> None:
        """Counts repetitions using a state machine driven by knee angle trends."""
        # On the first few frames, calibrate the standing angle
        if self.calibrated_standing_knee_angle is None:
            if len(self.knee_angle_history) > 5:
                self.calibrated_standing_knee_angle = np.mean(list(self.knee_angle_history))
                logging.info(f"Calibrated standing angle at frame {frame_idx}: {self.calibrated_standing_knee_angle:.2f}")
            return

        current_time = frame_idx / fps
        knee_trend = self.get_knee_angle_trend()

        standing_threshold = self.calibrated_standing_knee_angle * 0.95
        
        if self.rep_state == "standing":
            if knee_trend == "descending":
                self.rep_state = "descending"
                self.current_rep_start_frame = frame_idx
                self.current_rep_start_time = current_time

        elif self.rep_state == "descending":
            if knee_trend == "ascending":
                self.rep_state = "ascending"

        elif self.rep_state == "ascending":
            if avg_knee_angle >= standing_threshold:
                rep_duration = current_time - self.current_rep_start_time
                if self.MIN_REP_DURATION <= rep_duration <= self.MAX_REP_DURATION:
                    self.reps_completed += 1
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

    def detect_movement_phase(self, hip_height: float) -> bool:
        """Determine if in concentric (upward) phase based on hip movement."""
        if self.prev_hip_height is None:
            return False
        return hip_height > self.prev_hip_height
            
    def analyze_frame(self, landmarks, world_landmarks, frame_idx: int, fps: float, frame) -> Dict:
        """Analyze a single frame."""
        if not landmarks or not world_landmarks:
            return None
            
        normalized_points = self.normalize_pose(landmarks)
        angles = self.calculate_joint_angles(normalized_points)
        avg_knee_angle = (angles['left_knee'] + angles['right_knee']) / 2
        
        self.knee_angle_history.append(avg_knee_angle)

        left_hip_world = self.get_3d_point(world_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_world = self.get_3d_point(world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center_world = (left_hip_world + right_hip_world) / 2
        hip_height_world = hip_center_world[1]

        world_velocities = {}
        world_accelerations = {}
        
        body_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

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
        
        self.count_reps(avg_knee_angle, frame_idx, fps)
        
        is_concentric = self.detect_movement_phase(hip_height_world)
        
        intensity_value = 0
        frame_volume = 0
        hip_acceleration_magnitude = 0
        if world_accelerations:
             left_hip_acc = world_accelerations.get("LEFT_HIP", np.array([0,0,0]))
             right_hip_acc = world_accelerations.get("RIGHT_HIP", np.array([0,0,0]))
             hip_acceleration = (left_hip_acc + right_hip_acc) / 2
             hip_acceleration_magnitude = np.linalg.norm(hip_acceleration)
            
             if is_concentric:
                intensity_value = hip_acceleration_magnitude
             else:
                intensity_value = 1.0 / (1.0 + hip_acceleration_magnitude)
            
             self.avg_acceleration.append(intensity_value)
             self.max_acceleration = max(self.max_acceleration, intensity_value)

        if is_concentric and self.prev_hip_height is not None and self.user_weight > 0:
            vertical_distance = abs(self.prev_hip_height - hip_height_world)
            frame_volume = vertical_distance * self.user_weight
            self.total_volume += frame_volume

        self.accumulated_volume_over_time.append({
            'time': frame_idx / fps,
            'volume': self.total_volume
        })
        
        frame_data = {
            'frame_idx': frame_idx,
            'hip_height': hip_height_world,
            'hip_acceleration': hip_acceleration_magnitude,
            'is_concentric': is_concentric,
            'phase_intensity': intensity_value,
            'frame_volume': frame_volume,
            'accumulated_volume': self.total_volume,
            'is_analyzing': self.is_analyzing,
            'exercise_state': self.rep_state,
            'current_reps': self.reps_completed,
            'time': frame_idx / fps
        }
        
        for joint, angle in angles.items():
            frame_data[f"{joint}_angle"] = angle
        
        frame_data['velocities'] = {name: np.linalg.norm(vel) for name, vel in world_velocities.items()}
        frame_data['accelerations'] = {name: np.linalg.norm(acc) for name, acc in world_accelerations.items()}

        self.frame_metrics.append(frame_data)
        self.concentric_phase = is_concentric
        
        self.prev_knee_angle = avg_knee_angle
        self.prev_hip_height = hip_height_world
        self.prev_world_landmarks = world_landmarks
        self.prev_world_velocities = world_velocities

        self.draw_landmarks_with_state(frame, landmarks, self.rep_state, {})
        
        return frame_data

    def draw_landmarks_with_state(self, frame, landmarks, rep_state: str, rep_info: Dict):
        """Draw landmarks with different colors based on rep state."""
        if rep_state == "standing":
            line_color = (0, 255, 0)      # Green
            point_color = (255, 0, 0)     # Blue
        elif rep_state == "descending":
            line_color = (0, 255, 255)    # Yellow
            point_color = (0, 128, 255)   # Orange
        else:  # ascending
            line_color = (255, 255, 0)    # Cyan
            point_color = (255, 0, 255)   # Magenta
        
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
            cv2.circle(frame, start_pixel, 4, point_color, -1)
            cv2.circle(frame, end_pixel, 4, point_color, -1)

    def apply_smoothing(self, time_series):
        """Applies Savitzky-Golay filter to time-series data."""
        if not time_series:
            return time_series

        # First, find all possible keys by iterating through all frames
        all_keys = set()
        velocity_keys = set()
        acceleration_keys = set()
        for frame in time_series:
            all_keys.update(frame.keys())
            if 'velocities' in frame and isinstance(frame['velocities'], dict):
                velocity_keys.update(frame['velocities'].keys())
            if 'accelerations' in frame and isinstance(frame['accelerations'], dict):
                acceleration_keys.update(frame['accelerations'].keys())

        # Keys to smooth
        keys_to_smooth = [k for k in all_keys if k not in ['frame_idx', 'time', 'is_concentric', 'is_analyzing', 'exercise_state', 'velocities', 'accelerations']]

        # Prepare a dictionary to hold the lists of data for smoothing
        data_to_smooth = {key: [] for key in keys_to_smooth}
        data_to_smooth['velocities'] = {key: [] for key in velocity_keys}
        data_to_smooth['accelerations'] = {key: [] for key in acceleration_keys}

        # Populate the dictionary, filling missing values with 0
        for frame in time_series:
            for key in keys_to_smooth:
                data_to_smooth[key].append(frame.get(key, 0))
            for v_key in velocity_keys:
                data_to_smooth['velocities'][v_key].append(frame.get('velocities', {}).get(v_key, 0))
            for a_key in acceleration_keys:
                data_to_smooth['accelerations'][a_key].append(frame.get('accelerations', {}).get(a_key, 0))
        
        # Apply the filter
        window_length = min(11, len(time_series))
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length > 3:
            for key, values in data_to_smooth.items():
                if isinstance(values, dict):  # For velocities and accelerations
                    for sub_key, sub_values in values.items():
                        if len(sub_values) >= window_length:
                            data_to_smooth[key][sub_key] = savgol_filter(sub_values, window_length, 3).tolist()
                else:
                    if len(values) >= window_length:
                        data_to_smooth[key] = savgol_filter(values, window_length, 3).tolist()

        # Reconstruct the time_series with smoothed data
        smoothed_time_series = []
        for i in range(len(time_series)):
            new_frame = {
                'frame_idx': time_series[i]['frame_idx'],
                'time': time_series[i]['time'],
                'is_concentric': time_series[i].get('is_concentric'),
                'is_analyzing': time_series[i].get('is_analyzing'),
                'exercise_state': time_series[i].get('exercise_state'),
            }
            for key in keys_to_smooth:
                new_frame[key] = data_to_smooth[key][i]
            
            new_frame['velocities'] = {key: data_to_smooth['velocities'][key][i] for key in velocity_keys}
            new_frame['accelerations'] = {key: data_to_smooth['accelerations'][key][i] for key in acceleration_keys}
            
            smoothed_time_series.append(new_frame)
            
        return smoothed_time_series


    def get_final_analysis(self) -> Dict:
        """Get the final analysis."""
        if self.frame_metrics:
            last_frame = self.frame_metrics[-1]
            if self.analysis_start_frame is not None:
                analysis_start_time = self.analysis_start_frame / 30.0
                self.total_tension_time = last_frame['time'] - analysis_start_time

        avg_intensity = np.mean(self.avg_acceleration) if self.avg_acceleration else 0
        
        time_series = self.apply_smoothing(self.frame_metrics)
        
        # Flatten the velocities and accelerations
        for frame in time_series:
            if 'velocities' in frame:
                for joint, value in frame['velocities'].items():
                    frame[f'{joint.lower()}_velocity'] = value
                del frame['velocities']
            if 'accelerations' in frame:
                for joint, value in frame['accelerations'].items():
                    frame[f'{joint.lower()}_acceleration'] = value
                del frame['accelerations']

        time_series = [
            self.convert_numpy_types(frame) for frame in time_series
        ]
        
        volume_over_time = [
            {'time': float(p['time']), 'volume': float(p['volume'])} 
            for p in self.accumulated_volume_over_time
        ]
        
        volume_score = min(100, (self.total_volume / 15.0) * 100) if self.total_volume > 0 else 0
        tut_score = min(100, (self.total_tension_time / 45.0) * 100) if self.total_tension_time > 0 else 0
        intensity_score = min(100, avg_intensity * 50) if avg_intensity > 0 else 0
        total_score = (volume_score * 0.4) + (intensity_score * 0.3) + (tut_score * 0.3)
        
        avg_rep_time = 0
        if self.rep_details:
            total_rep_duration = sum(rep['duration'] for rep in self.rep_details)
            avg_rep_time = total_rep_duration / len(self.rep_details)
        
        time_efficiency = 0
        if self.frame_metrics and self.total_tension_time > 0:
            total_video_time = self.frame_metrics[-1]['time'] - self.frame_metrics[0]['time']
            if total_video_time > 0:
                time_efficiency = (self.total_tension_time / total_video_time) * 100

        analysis_results = {
            'scores': {'overall': total_score, 'intensity': intensity_score, 'tut': tut_score, 'volume': volume_score},
            'reps': {'total': self.reps_completed, 'avg_duration': avg_rep_time, 'details': self.rep_details},
            'metrics': {
                'time_under_tension': self.total_tension_time,
                'time_efficiency': time_efficiency,
                'total_volume': {'value': self.total_volume, 'unit': 'kg·m'},
                'max_intensity': self.max_acceleration,
                'avg_intensity': avg_intensity,
            },
            'time_series_data': {'volume_progression': volume_over_time, 'kinematics': time_series}
        }
        
        return self.convert_numpy_types(analysis_results)

    def convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for compatibility."""
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
