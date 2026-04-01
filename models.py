from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ModerationAction(str, Enum):
    ALLOW = "ALLOW"
    REMOVE = "REMOVE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    WARN_USER = "WARN_USER"
    SHADOW_BAN = "SHADOW_BAN"

class UserHistory(BaseModel):
    user_id: str
    violations_count: int = 0
    trust_score: float = 1.0  # 0.0 to 1.0
    previous_actions: List[ModerationAction] = []

class PostContext(BaseModel):
    thread_id: str
    related_post_ids: List[str] = []
    parent_post_id: Optional[str] = None

class PolicyRule(BaseModel):
    category: str
    description: str
    severity: str

class Observation(BaseModel):
    post_id: str
    content: str
    user_history: UserHistory
    context: PostContext
    policy_rules: List[PolicyRule]

class Action(BaseModel):
    decision: ModerationAction
    reasoning: str = Field(..., description="Justification for the moderation decision")

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float] = {}

class GroundTruth(BaseModel):
    post_id: str
    label: str  # One of: Hate Speech, Harassment, Self-Harm, Scam/Fraud, Misinformation, Safe
    is_violation: bool
    explanation: str

class State(BaseModel):
    moderation_queue: List[Observation] = []
    processed_decisions: List[Dict[str, Any]] = []
    user_db: Dict[str, UserHistory] = {}
    ground_truth: Dict[str, GroundTruth] = {}
    current_step: int = 0
    total_steps: int = 0
