from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, State, ModerationAction, GroundTruth
from data_engine import DataEngine

class ContentModerationEnv:
    def __init__(self, difficulty: str = "EASY", max_steps: int = 10, seed: int = 42):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.data_engine = DataEngine(seed=seed)
        self.state_data = State(total_steps=max_steps)
        self._initialize_queue()

    def _initialize_queue(self):
        self.state_data.moderation_queue = []
        self.state_data.ground_truth = {}
        for _ in range(self.max_steps):
            obs, gt = self.data_engine.generate_scenario(self.difficulty)
            self.state_data.moderation_queue.append(obs)
            self.state_data.ground_truth[obs.post_id] = gt

    def reset(self) -> Observation:
        self.state_data.current_step = 0
        self.state_data.processed_decisions = []
        # Re-initialize for fresh start if needed, but here we just return the first
        return self.state_data.moderation_queue[0]

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.state_data.current_step >= self.max_steps:
            return None, Reward(value=0), True, {}

        current_obs = self.state_data.moderation_queue[self.state_data.current_step]
        gt = self.state_data.ground_truth[current_obs.post_id]
        
        # Calculate Reward
        reward = self._calculate_reward(action, gt)
        
        # Update State
        self.state_data.processed_decisions.append({
            "post_id": current_obs.post_id,
            "action": action.decision,
            "reward": reward.value,
            "ground_truth": gt.label
        })
        
        # Update User History (Trust Score Evolution)
        user_id = current_obs.user_history.user_id
        user = self.data_engine.user_db[user_id]
        if action.decision == ModerationAction.REMOVE and gt.is_violation:
            user.violations_count += 1
            user.trust_score = max(0.0, user.trust_score - 0.2)
        elif action.decision == ModerationAction.ALLOW and not gt.is_violation:
            user.trust_score = min(1.0, user.trust_score + 0.05)
        
        self.state_data.current_step += 1
        done = self.state_data.current_step >= self.max_steps
        
        next_obs = None
        if not done:
            next_obs = self.state_data.moderation_queue[self.state_data.current_step]
            # Update user history in the next observation if it's the same user
            if next_obs.user_history.user_id == user_id:
                next_obs.user_history = user

        return next_obs, reward, done, {"ground_truth": gt.dict()}

    def _calculate_reward(self, action: Action, gt: GroundTruth) -> Reward:
        val = 0.0
        breakdown = {}

        decision = action.decision
        is_violation = gt.is_violation

        if decision == ModerationAction.REMOVE:
            if is_violation:
                val = 1.0
                breakdown["correct_remove"] = 1.0
            else:
                val = -0.7
                breakdown["false_positive"] = -0.7
        elif decision == ModerationAction.ALLOW:
            if not is_violation:
                val = 0.8
                breakdown["correct_allow"] = 0.8
            else:
                val = -1.0
                breakdown["false_negative_dangerous"] = -1.0
        elif decision == ModerationAction.FLAG_FOR_REVIEW:
            val = 0.5
            breakdown["safe_uncertainty"] = 0.5
        elif decision == ModerationAction.WARN_USER:
            if is_violation:
                val = 0.6
                breakdown["warn_violation"] = 0.6
            else:
                val = -0.3
                breakdown["warn_safe"] = -0.3
        elif decision == ModerationAction.SHADOW_BAN:
            if is_violation:
                val = 0.9
                breakdown["ban_violation"] = 0.9
            else:
                val = -0.8
                breakdown["ban_safe"] = -0.8

        # Shaping: Context reward (simplified: if reasoning mentions history/context/rules)
        if "history" in action.reasoning.lower() or "previous" in action.reasoning.lower():
            val += 0.1
            breakdown["context_bonus"] = 0.1

        return Reward(value=val, breakdown=breakdown)

    def state(self) -> State:
        return self.state_data
