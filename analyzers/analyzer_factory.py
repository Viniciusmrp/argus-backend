from analyzers.squat_analyzer import SquatAnalyzer

def get_analyzer(exercise_type: str):
    """
    Factory function to get an analyzer based on exercise type.
    """
    normalized_exercise = exercise_type.lower()

    if normalized_exercise in ["squat", "back squat"]:
        return SquatAnalyzer()
    # Add other exercises here
    # elif normalized_exercise == "deadlift":
    #     return DeadliftAnalyzer()
    else:
        raise ValueError(f"Unknown exercise type: {exercise_type}")
