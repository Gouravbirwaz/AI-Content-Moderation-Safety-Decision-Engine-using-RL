from abc import ABC, abstractmethod
from typing import List, Dict, Any
from models import Observation, Action, GroundTruth, ModerationAction

class TaskGrader(ABC):
    @abstractmethod
    def score(self, actions: List[Action], ground_truth_list: List[GroundTruth]) -> float:
        pass

class ModerationGrader(TaskGrader):
    def score(self, actions: List[Action], ground_truth_list: List[GroundTruth]) -> float:
        if not actions:
            return 0.0
        
        total_items = len(ground_truth_list)
        individual_scores = []

        for action, gt in zip(actions, ground_truth_list):
            decision = action.decision
            is_violation = gt.is_violation
            
            # Grader Score Calculation (0.0 to 1.0)
            if decision == ModerationAction.REMOVE:
                if is_violation:
                    individual_scores.append(1.0)  # Correct REMOVE
                else:
                    individual_scores.append(0.3)  # False Positive (heavy penalty vs 1.0)
            elif decision == ModerationAction.ALLOW:
                if not is_violation:
                    individual_scores.append(1.0)  # Correct ALLOW
                else:
                    individual_scores.append(0.0)  # False Negative (most dangerous)
            elif decision == ModerationAction.FLAG_FOR_REVIEW:
                individual_scores.append(0.8 if is_violation else 0.6) # Reward safe uncertainty
            elif decision == ModerationAction.WARN_USER:
                individual_scores.append(0.9 if is_violation else 0.4) # Slightly less than REMOVE/ALLOW
            elif decision == ModerationAction.SHADOW_BAN:
                individual_scores.append(1.0 if is_violation else 0.2) # High penalty for shadow-banning safe users
        
        return sum(individual_scores) / total_items

class TaskDefinition:
    def __init__(self, name: str, difficulty: str, max_steps: int, description: str):
        self.name = name
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.description = description
        self.grader = ModerationGrader()

TASKS = [
    TaskDefinition(
        name="Easy Moderation",
        difficulty="EASY",
        max_steps=5,
        description="Handle obvious violations like spam and explicit hate speech."
    ),
    TaskDefinition(
        name="Medium Moderation",
        difficulty="MEDIUM",
        max_steps=10,
        description="Handle borderline cases, sarcasm, and context-dependent flags."
    ),
    TaskDefinition(
        name="Hard Moderation",
        difficulty="HARD",
        max_steps=15,
        description="Reason through coordinated adversarial attempts and multi-step misinformation."
    )
]
