from analyzers.squat_analyzer import SquatAnalyzer

def get_analyzer(exercise_type: str):
    """
    Factory function to get an analyzer based on exercise type.
    """
    if exercise_type == "squat":
        return SquatAnalyzer()
    # Add other exercises here
    # elif exercise_type == "deadlift":
    #     return DeadliftAnalyzer()
    else:
        raise ValueError(f"Unknown exercise type: {exercise_type}")
