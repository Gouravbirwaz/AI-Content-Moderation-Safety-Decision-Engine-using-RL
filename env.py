from typing import Tuple, Dict, Any, List
from models import Observation, Action, Reward, State, ModerationAction, GroundTruth, UserHistory
from data_engine import DataEngine

class ContentModerationEnv:
    def __init__(self, difficulty: str = "EASY", max_steps: int = 10, seed: int = 42, is_sequential: bool = False):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.is_sequential = is_sequential
        self.data_engine = DataEngine(seed=seed)
        self.state_data = State(total_steps=max_steps, platform_risk_score=0.0, policy_version="1.0")
        self._initialize_queue()

    def _initialize_queue(self):
        self.state_data.moderation_queue = []
        self.state_data.ground_truth = {}
        
        if self.is_sequential:
            scenarios = self.data_engine.generate_sequential_scenarios(self.max_steps)
            for obs, gt in scenarios:
                obs.platform_risk_score = self.state_data.platform_risk_score
                obs.policy_version = self.state_data.policy_version
                self.state_data.moderation_queue.append(obs)
                self.state_data.ground_truth[obs.post_id] = gt
        else:
            for _ in range(self.max_steps):
                obs, gt = self.data_engine.generate_scenario(self.difficulty)
                obs.platform_risk_score = self.state_data.platform_risk_score
                obs.policy_version = self.state_data.policy_version
                self.state_data.moderation_queue.append(obs)
                self.state_data.ground_truth[obs.post_id] = gt

    def reset(self) -> Observation:
        self.state_data.current_step = 0
        self.state_data.processed_decisions = []
        self.state_data.platform_risk_score = 0.0
        self.state_data.policy_version = "1.0"
        return self.state_data.moderation_queue[0]

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state_data.current_step >= self.max_steps:
            return None, Reward(value=0), True, {}

        current_obs = self.state_data.moderation_queue[self.state_data.current_step]
        gt = self.state_data.ground_truth[current_obs.post_id]
        
        # Policy Shift Logic (Switch to 2.0 halfway)
        if self.state_data.current_step == self.max_steps // 2:
            self.state_data.policy_version = "2.0 (STRICT)"
        
        # Calculate Risk Delta
        risk_delta = self._calculate_risk_delta(action, gt, current_obs.user_history)
        self.state_data.platform_risk_score = max(0.0, self.state_data.platform_risk_score + risk_delta)
        
        # Calculate Reward
        reward = self._calculate_reward(action, gt, current_obs.user_history)
        
        # Update State
        self.state_data.processed_decisions.append({
            "post_id": current_obs.post_id,
            "action": action.decision,
            "reward": reward.value,
            "ground_truth": gt.label,
            "risk_score": self.state_data.platform_risk_score
        })
        
        # Update User History (Persistence)
        user_id = current_obs.user_history.user_id
        user = self.data_engine.user_db[user_id]
        
        # Track history
        user.history.append({
            "post_id": current_obs.post_id,
            "content": current_obs.content,
            "decision": action.decision,
            "is_violation": gt.is_violation
        })
        
        if action.decision in [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN] and gt.is_violation:
            user.violations_count += 1
            user.trust_score = max(0.0, user.trust_score - 0.2)
        elif action.decision == ModerationAction.ALLOW and not gt.is_violation:
            user.trust_score = min(1.0, user.trust_score + 0.05)
        
        self.state_data.current_step += 1
        done = self.state_data.current_step >= self.max_steps
        
        next_obs = None
        if not done:
            next_obs = self.state_data.moderation_queue[self.state_data.current_step]
            # Refresh context for next observation
            next_obs.platform_risk_score = self.state_data.platform_risk_score
            next_obs.policy_version = self.state_data.policy_version
            
            # Update user history in the next observation if it's the same user
            if next_obs.user_history.user_id == user_id:
                next_obs.user_history = user

        return next_obs, reward, done, {"ground_truth": gt.model_dump(), "platform_risk": self.state_data.platform_risk_score}

    def _calculate_risk_delta(self, action: Action, gt: GroundTruth, user: UserHistory) -> float:
        """Industry-standard risk engine. Calculates impact on platform safety."""
        delta = 0.0
        
        # Severe Risk: False Negatives (Missed violations)
        if not (action.decision in [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN, ModerationAction.WARN_USER]) and gt.is_violation:
            delta += 0.3
            # Escalation: Repeat offenders post a systemic risk
            if user.violations_count > 0:
                delta += 0.2
            # Critical Risk: Self-harm or severe Hate Speech
            if gt.label in ["Self-Harm", "Hate Speech"]:
                delta += 0.2
                
        # Operational Risk: False Positives (Over-moderation)
        elif action.decision in [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN] and not gt.is_violation:
            delta += 0.1
            
        # Risk Mitigation: Correct enforcement
        elif (action.decision in [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN] and gt.is_violation) or \
             (action.decision == ModerationAction.ALLOW and not gt.is_violation):
            delta -= 0.2
        
        return delta

    def _calculate_reward(self, action: Action, gt: GroundTruth, user: UserHistory) -> Reward:
        """Dense multi-factor reward function for high-fidelity reinforcement learning."""
        val = 0.0
        breakdown = {}

        decision = action.decision
        is_violation = gt.is_violation

        # 1. Enforcement Accuracy (Primary Factor)
        if is_violation:
            if decision in [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN]:
                val += 1.5 # High reward for stopping violations
                breakdown["enforcement_success"] = 1.5
            elif decision == ModerationAction.WARN_USER:
                val += 0.8 # Partial reward for warning
                breakdown["preventative_action"] = 0.8
            else:
                val -= 2.0 # Severe penalty for missing a violation (FN)
                breakdown["safety_failure_penalty"] = -2.0
        else:
            if decision == ModerationAction.ALLOW:
                val += 1.0 # Reward for correctly allowing safe content
                breakdown["operational_efficiency"] = 1.0
            elif decision in [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN]:
                val -= 1.2 # Penalty for false positives (over-moderation)
                breakdown["over_moderation_penalty"] = -1.2
        
        # 2. Contextual reasoning bonus (Industry standard: Explainability)
        reasoning = action.reasoning.lower()
        if any(word in reasoning for word in ["history", "repeat", "violation", "past", "pattern"]):
            val += 0.3
            breakdown["historical_context_bonus"] = 0.3
            
        if any(word in reasoning for word in ["risk", "platform", "safety", "standard"]):
            val += 0.2
            breakdown["risk_awareness_bonus"] = 0.2

        # 3. Systemic Risk Penalty
        # Penalty grows non-linearly with platform risk
        risk_penalty = (self.state_data.platform_risk_score ** 2) * 0.2
        val -= risk_penalty
        breakdown["systemic_risk_drag"] = -risk_penalty

        return Reward(value=val, breakdown=breakdown)

    def state(self) -> State:
        return self.state_data
