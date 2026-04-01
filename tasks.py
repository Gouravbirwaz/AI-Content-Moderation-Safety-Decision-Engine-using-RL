from abc import ABC, abstractmethod
from typing import List, Dict, Any
from models import Observation, Action, GroundTruth, ModerationAction, State

class TaskGrader(ABC):
    @abstractmethod
    def score(self, actions: List[Action], ground_truth_list: List[GroundTruth], final_state: State) -> Dict[str, float]:
        pass

class MultiDimensionalGrader(TaskGrader):
    def score(self, actions: List[Action], ground_truth_list: List[GroundTruth], final_state: State) -> Dict[str, float]:
        if not actions:
            return {"final_score": 0.0}
        
        total_items = len(ground_truth_list)
        
        tp = 0 # True Positives (Action taken on violation)
        fp = 0 # False Positives (Action taken on safe content)
        tn = 0 # True Negatives (No action on safe content)
        fn = 0 # False Negatives (No action on violation)
        
        action_mapping = [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN, ModerationAction.WARN_USER]
        
        for action, gt in zip(actions, ground_truth_list):
            is_action_taken = action.decision in action_mapping
            if gt.is_violation:
                if is_action_taken:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_action_taken:
                    fp += 1
                else:
                    tn += 1
        
        # Calculate Industry Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_items
        
        # Risk Mitigation Score
        # Normalized based on total possible risk increase
        max_risk_possible = total_items * 0.5
        risk_mitigation = max(0.0, 1.0 - (final_state.platform_risk_score / max_risk_possible))
        
        # Policy Alignment (Strictness check)
        policy_adherence = 0
        for i, action in enumerate(actions):
            # If in strict mode (Version 2.0), we expect no ALLOW for borderline content
            is_strict = final_state.policy_version.startswith("2.0") and i >= total_items // 2
            if is_strict and action.decision == ModerationAction.ALLOW and ground_truth_list[i].label in ["Harassment", "Hate Speech"]:
                continue
            policy_adherence += 1
        policy_score = policy_adherence / total_items

        # Weighted Industry Score
        final_score = (f1_score * 0.4) + (risk_mitigation * 0.3) + (policy_score * 0.2) + (accuracy * 0.1)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "risk_mitigation": risk_mitigation,
            "policy_alignment": policy_score,
            "final_score": final_score
        }

class TaskDefinition:
    def __init__(self, name: str, difficulty: str, max_steps: int, description: str, is_sequential: bool = False):
        self.name = name
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.description = description
        self.is_sequential = is_sequential
        self.grader = MultiDimensionalGrader()

TASKS = [
    TaskDefinition(
        name="Dynamic Risk Management",
        difficulty="MEDIUM",
        max_steps=10,
        description="Handle a mix of safe and violating content while managing global platform risk."
    ),
    TaskDefinition(
        name="Sequential Adversarial Escalation",
        difficulty="HARD",
        max_steps=5,
        is_sequential=True,
        description="A single user escalating from safe behavior to sophisticated adversarial hate speech."
    ),
    TaskDefinition(
        name="Policy Shift Adaptation",
        difficulty="MEDIUM",
        max_steps=10,
        description="The moderation policy becomes stricter halfway through. Adapt or fail."
    )
]
