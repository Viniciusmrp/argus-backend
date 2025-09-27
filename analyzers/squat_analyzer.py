from collections import deque
import logging
import cv2
import numpy as np
from scipy.signal import savgol_filter
from analyzers.base_analyzer import BaseAnalyzer
from typing import Dict, List

class SquatAnalyzer(BaseAnalyzer):
    """
    An analyzer for squat exercises, focusing on rep counting, kinematic analysis,
    and performance metrics. Inherits from the generic BaseAnalyzer.
    """
    def __init__(self):
        super().__init__()
        
        # Rep counting parameters
        self.MIN_REP_DURATION = 0.8
        self.MAX_REP_DURATION = 8.0
        
        # State tracking for rep counting
        self.knee_angle_history = deque(maxlen=10)
        self.rep_state = "standing"  # "standing", "descending", "ascending"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.reps_completed = 0
        self.rep_details = []
        self.calibrated_standing_knee_angle = None
        
        # Metrics tracking
        self.is_analyzing = True
        self.analysis_start_frame = 0
        self.start_time = 0
        self.total_tension_time = 0
        self.tut_eccentric = 0
        self.tut_concentric = 0
        self.accumulated_volume_over_time = []
        self.max_acceleration = 0
        self.concentric_intensity = []
        self.eccentric_intensity = []
        self.total_intensity = 0
        
        # Previous state for specific metrics
        self.prev_hip_height = None
        
    def get_body_landmarks(self) -> List[int]:
        """Return the list of MediaPipe PoseLandmark indices relevant to squats."""
        return [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]

    def calculate_joint_angles(self, normalized_points: Dict[int, np.ndarray]) -> Dict[str, float]:
        """Calculate knee, hip, and ankle angles relevant for squats."""
        angles = {}
        
        # Knee angles
        angles['left_knee'] = self.calculate_3d_angle(
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_HIP),
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_KNEE),
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        )
        angles['right_knee'] = self.calculate_3d_angle(
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_HIP),
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_KNEE),
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        )
        
        # Hip angles
        angles['left_hip'] = self.calculate_3d_angle(
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_HIP),
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_KNEE)
        )
        angles['right_hip'] = self.calculate_3d_angle(
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_HIP),
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_KNEE)
        )
        
        # Ankle angles
        angles['left_ankle'] = self.calculate_3d_angle(
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_KNEE),
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_ANKLE),
            normalized_points.get(self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        )
        angles['right_ankle'] = self.calculate_3d_angle(
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_KNEE),
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            normalized_points.get(self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        )
        
        return angles

    def reset_analysis(self):
        """Reset analysis state for a new video, including squat-specific state."""
        super().reset_analysis()
        self.knee_angle_history.clear()
        self.rep_state = "standing"
        self.current_rep_start_frame = None
        self.current_rep_start_time = None
        self.reps_completed = 0
        self.rep_details = []
        self.calibrated_standing_knee_angle = None
        self.is_analyzing = True
        self.analysis_start_frame = 0
        self.start_time = 0
        self.total_tension_time = 0
        self.tut_eccentric = 0
        self.tut_concentric = 0
        self.accumulated_volume_over_time = []
        self.max_acceleration = 0
        self.concentric_intensity = []
        self.eccentric_intensity = []
        self.total_intensity = 0
        self.prev_hip_height = None

    def get_knee_angle_trend(self) -> str:
        """Determines the trend of knee angle movement (ascending, descending, stationary)."""
        if len(self.knee_angle_history) < 5:
            return "stationary"
        x = np.arange(len(self.knee_angle_history))
        y = np.array(self.knee_angle_history)
        slope, _ = np.polyfit(x, y, 1)

        if abs(slope) < 0.5: return "stationary"
        return "ascending" if slope > 0 else "descending"

    def count_reps(self, avg_knee_angle: float, frame_idx: int, fps: float):
        """Counts squat repetitions using a state machine driven by knee angle trends."""
        if self.calibrated_standing_knee_angle is None:
            if len(self.knee_angle_history) > 5:
                self.calibrated_standing_knee_angle = np.mean(list(self.knee_angle_history))
            return

        current_time = frame_idx / fps
        knee_trend = self.get_knee_angle_trend()
        standing_threshold = self.calibrated_standing_knee_angle * 0.95
        
        if self.rep_state == "standing" and knee_trend == "descending":
            self.rep_state = "descending"
            self.current_rep_start_frame = frame_idx
            self.current_rep_start_time = current_time
        elif self.rep_state == "descending" and knee_trend == "ascending":
            self.rep_state = "ascending"
        elif self.rep_state == "ascending" and avg_knee_angle >= standing_threshold:
            rep_duration = current_time - self.current_rep_start_time
            if self.MIN_REP_DURATION <= rep_duration <= self.MAX_REP_DURATION:
                self.reps_completed += 1
                self.rep_details.append({
                    'rep_number': self.reps_completed,
                    'start_frame': self.current_rep_start_frame, 'end_frame': frame_idx,
                    'duration': rep_duration,
                    'start_time': self.current_rep_start_time, 'end_time': current_time
                })
            self.rep_state = "standing"

    def detect_movement_phase(self, hip_height: float) -> bool:
        """Determine if in concentric (upward) phase based on hip movement."""
        is_concentric = hip_height > self.prev_hip_height if self.prev_hip_height is not None else False
        return is_concentric
            
    def process_exercise_specific_metrics(self, frame_idx: int, fps: float, normalized_points: Dict, angles: Dict, 
                                          world_velocities: Dict, world_accelerations: Dict, world_landmarks) -> Dict:
        """Process squat-specific metrics like rep counting and volume."""
        avg_knee_angle = (angles.get('left_knee', 180) + angles.get('right_knee', 180)) / 2
        self.knee_angle_history.append(avg_knee_angle)

        left_hip_world = self.get_3d_point(world_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip_world = self.get_3d_point(world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        hip_center_world = (left_hip_world + right_hip_world) / 2
        hip_height_world = hip_center_world[1]

        self.count_reps(avg_knee_angle, frame_idx, fps)
        
        is_concentric = self.detect_movement_phase(hip_height_world)
        
        frame_duration = 1.0 / fps
        if self.rep_state == "descending" or self.rep_state == "ascending":
            if is_concentric:
                self.tut_concentric += frame_duration
            else:
                self.tut_eccentric += frame_duration
        
        hip_acceleration_magnitude = 0
        phase_intensity = 0
        concentric_intensity_value = 0
        eccentric_intensity_value = 0
        
        if world_accelerations:
             left_hip_acc = world_accelerations.get("LEFT_HIP", np.array([0,0,0]))
             right_hip_acc = world_accelerations.get("RIGHT_HIP", np.array([0,0,0]))
             hip_acceleration = (left_hip_acc + right_hip_acc) / 2
             hip_acceleration_magnitude = np.linalg.norm(hip_acceleration)
            
             if is_concentric:
                 concentric_intensity_value = hip_acceleration_magnitude
                 self.concentric_intensity.append(concentric_intensity_value)
                 phase_intensity = concentric_intensity_value
             else:
                 eccentric_intensity_value = 1.0 / (1.0 + hip_acceleration_magnitude)
                 self.eccentric_intensity.append(eccentric_intensity_value)
                 phase_intensity = eccentric_intensity_value
             
             self.total_intensity += phase_intensity
             self.max_acceleration = max(self.max_acceleration, hip_acceleration_magnitude)

        frame_volume = 0
        if is_concentric and self.prev_hip_height is not None and self.user_weight > 0:
            vertical_distance = abs(self.prev_hip_height - hip_height_world)
            frame_volume = vertical_distance * self.user_weight
            self.total_volume += frame_volume

        self.accumulated_volume_over_time.append({'time': frame_idx / fps, 'volume': self.total_volume})
        
        frame_data = {
            'frame_idx': frame_idx, 'time': frame_idx / fps,
            'hip_height': hip_height_world, 'hip_acceleration': hip_acceleration_magnitude,
            'is_concentric': is_concentric, 'phase_intensity': phase_intensity,
            'concentric_intensity': concentric_intensity_value,
            'eccentric_intensity': eccentric_intensity_value,
            'frame_volume': frame_volume, 'accumulated_volume': self.total_volume,
            'is_analyzing': self.is_analyzing, 'exercise_state': self.rep_state,
            'current_reps': self.reps_completed,
            'velocities': {name: np.linalg.norm(vel) for name, vel in world_velocities.items()},
            'accelerations': {name: np.linalg.norm(acc) for name, acc in world_accelerations.items()}
        }
        frame_data.update({f"{joint}_angle": angle for joint, angle in angles.items()})
        
        self.prev_hip_height = hip_height_world
        return frame_data

    def draw_landmarks(self, frame, landmarks, frame_data: Dict):
        """Draw landmarks with colors based on the current repetition state."""
        rep_state = frame_data.get('exercise_state', 'standing')
        state_colors = {
            "standing": ((0, 255, 0), (255, 0, 0)),
            "descending": ((0, 255, 255), (0, 128, 255)),
            "ascending": ((255, 255, 0), (255, 0, 255))
        }
        line_color, point_color = state_colors.get(rep_state, state_colors["standing"])
        
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        h, w = frame.shape[:2]
        for start_idx, end_idx in connections:
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            start_pixel = (int(start_point.x * w), int(start_point.y * h))
            end_pixel = (int(end_point.x * w), int(end_point.y * h))
            cv2.line(frame, start_pixel, end_pixel, line_color, 2)
            cv2.circle(frame, start_pixel, 4, point_color, -1)
            cv2.circle(frame, end_pixel, 4, point_color, -1)

    def apply_smoothing(self, time_series):
        """Applies Savitzky-Golay filter to smooth the time-series data."""
        if not time_series: return []

        all_keys = {k for frame in time_series for k in frame.keys()}
        velocity_keys = {k for frame in time_series if 'velocities' in frame for k in frame['velocities']}
        acceleration_keys = {k for frame in time_series if 'accelerations' in frame for k in frame['accelerations']}
        
        keys_to_smooth = [k for k in all_keys if k not in ['frame_idx', 'time', 'is_concentric', 'is_analyzing', 'exercise_state', 'velocities', 'accelerations']]
        
        data = {key: [f.get(key, 0) for f in time_series] for key in keys_to_smooth}
        data['velocities'] = {key: [f.get('velocities', {}).get(key, 0) for f in time_series] for key in velocity_keys}
        data['accelerations'] = {key: [f.get('accelerations', {}).get(key, 0) for f in time_series] for key in acceleration_keys}

        window_length = min(11, len(time_series))
        if window_length > 3 and window_length % 2 != 0:
            for key, values in data.items():
                if isinstance(values, dict):
                    for sub_key, sub_values in values.items():
                        if len(sub_values) >= window_length: data[key][sub_key] = savgol_filter(sub_values, window_length, 3).tolist()
                elif len(values) >= window_length:
                    data[key] = savgol_filter(values, window_length, 3).tolist()

        smoothed_series = []
        for i, frame in enumerate(time_series):
            new_frame = {k: frame.get(k) for k in ['frame_idx', 'time', 'is_concentric', 'is_analyzing', 'exercise_state']}
            new_frame.update({key: data[key][i] for key in keys_to_smooth})
            new_frame['velocities'] = {key: data['velocities'][key][i] for key in velocity_keys}
            new_frame['accelerations'] = {key: data['accelerations'][key][i] for key in acceleration_keys}
            smoothed_series.append(new_frame)
            
        return smoothed_series

    def get_final_analysis(self) -> Dict:
        """Compile and return the final analysis summary."""
        if self.frame_metrics:
            self.total_tension_time = self.tut_concentric + self.tut_eccentric

        time_series = self.apply_smoothing(self.frame_metrics)

        # Flatten nested kinematics for easier consumption
        for frame in time_series:
            for joint, value in frame.pop('velocities', {}).items(): frame[f'{joint.lower()}_velocity'] = value
            for joint, value in frame.pop('accelerations', {}).items(): frame[f'{joint.lower()}_acceleration'] = value
        
        avg_concentric_intensity = np.mean(self.concentric_intensity) if self.concentric_intensity else 0
        avg_eccentric_intensity = np.mean(self.eccentric_intensity) if self.eccentric_intensity else 0
        
        concentric_intensity_score = min(100, avg_concentric_intensity * 50)
        eccentric_intensity_score = min(100, avg_eccentric_intensity * 100)
        
        analysis = {
            'scores': {
                'concentric_intensity': concentric_intensity_score,
                'eccentric_intensity': eccentric_intensity_score
            },
            'reps': {
                'total': self.reps_completed,
                'avg_duration': np.mean([rep['duration'] for rep in self.rep_details]) if self.rep_details else 0,
                'details': self.rep_details
            },
            'metrics': {
                'time_under_tension': self.total_tension_time,
                'tut_concentric': self.tut_concentric,
                'tut_eccentric': self.tut_eccentric,
                'time_efficiency': (self.total_tension_time / (self.frame_metrics[-1]['time'] - self.frame_metrics[0]['time']) * 100) if self.frame_metrics and self.frame_metrics[-1]['time'] > self.frame_metrics[0]['time'] else 0,
                'total_volume': {'value': self.total_volume, 'unit': 'kg·m'},
                'max_intensity': self.max_acceleration, 
                'avg_concentric_intensity': avg_concentric_intensity,
                'avg_eccentric_intensity': avg_eccentric_intensity,
                'total_intensity': self.total_intensity,
            },
            'time_series_data': {
                'volume_progression': self.accumulated_volume_over_time,
                'kinematics': self.convert_numpy_types(time_series)
            }
        }
        return self.convert_numpy_types(analysis)
