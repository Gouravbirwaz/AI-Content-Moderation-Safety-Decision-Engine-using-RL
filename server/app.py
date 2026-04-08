import os
import sys
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from env.env import ContentModerationEnv
from env.models import Observation, Action, Reward, State, ModerationAction
from tasks.tasks import TASKS

app = FastAPI(title="AI Content Moderation OpenEnv API")

def main():
    """Entry point for the server script."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()

# Global environment state (simplified for hackathon single-session evaluation)
current_env: Optional[ContentModerationEnv] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = "basic_safety"

class StepRequest(BaseModel):
    action: Action

@app.get("/")
def read_root():
    return {
        "status": "online",
        "engine": "AI Multimodal Moderation",
        "version": "1.1.0",
        "endpoints": ["/reset", "/step", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    global current_env
    
    # Default parameters
    difficulty = "MEDIUM"
    max_steps = 10
    is_sequential = False
    has_images = True
    
    task_id = request.task_id if request else "basic_safety"
    
    # Map task slugs to TaskDefinitions
    # Common hackathon patterns use slugs like 'multimodal_hard'
    task_map = {
        "basic_safety": TASKS[0],
        "multimodal_hard": TASKS[4], # Multimodal Adversarial Evasion
        "policy_drift": TASKS[3],    # Policy Shift Adaptation
        "dynamic_risk": TASKS[1],
        "sequential_hate": TASKS[2],
        "visual_escalation": TASKS[5]
    }
    
    task_def = task_map.get(task_id, TASKS[1]) # Default to dynamic_risk
    
    current_env = ContentModerationEnv(
        difficulty=task_def.difficulty,
        max_steps=task_def.max_steps,
        is_sequential=task_def.is_sequential,
        has_images=task_def.has_images
    )
    
    obs = current_env.reset()
    return obs.model_dump()

@app.post("/step")
async def step(request: StepRequest):
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        obs, reward, done, info = current_env.step(request.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state():
    if current_env is None:
        return {"error": "Not initialized"}
    return current_env.state().model_dump()



