import os
import json
import asyncio
from typing import List
import google.generativeai as genai
from env import ContentModerationEnv
from tasks import TASKS
from models import Action, ModerationAction

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY", "your-api-key")
model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")

if api_key and api_key != "your-api-key":
    genai.configure(api_key=api_key)

SYSTEM_PROMPT = """You are a senior Content Moderation agent. Your goal is to make policy-aligned decisions on user-generated content.
Policy Categories: Hate Speech, Harassment, Self-Harm, Scam/Fraud, Misinformation, Adversarial.
Available Actions:
- ALLOW: Safe content.
- REMOVE: Content that violates policy.
- FLAG_FOR_REVIEW: Borderline/Ambiguous cases.
- WARN_USER: Minor violations or first-time warnings.
- SHADOW_BAN: Severe, repetitive, or adversarial violations.

Output your decision in strict JSON format:
{
  "decision": "ACTION_NAME",
  "reasoning": "Brief explanation of the decision including context and user history."
}
"""

async def run_task(task_def):
    print(f"\n--- Running Task: {task_def.name} ---")
    env = ContentModerationEnv(difficulty=task_def.difficulty, max_steps=task_def.max_steps)
    obs = env.reset()
    done = False
    
    actions = []
    ground_truth_list = []
    
    # Initialize Gemini model
    model = None
    if api_key and api_key != "your-api-key":
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"}
        )
    
    while not done:
        prompt = f"Content: {obs.content}\nUser History: {obs.user_history.dict()}\nRules: {[r.dict() for r in obs.policy_rules]}"
        
        if not model:
            # Dummy moderation logic for verification without API key
            decision = ModerationAction.REMOVE if "violation" in obs.content.lower() or "hate" in obs.content.lower() else ModerationAction.ALLOW
            decision_data = {"decision": decision, "reasoning": "Dummy decision for verification."}
        else:
            response = model.generate_content(prompt)
            decision_data = json.loads(response.text)
        
        action = Action(**decision_data)
        
        next_obs, reward, done, info = env.step(action)
        
        actions.append(action)
        ground_truth_list.append(env.state().ground_truth[obs.post_id])
        
        print(f"Post: {obs.content[:50]}... -> Action: {action.decision} | Reward: {reward.value}")
        obs = next_obs

    final_score = task_def.grader.score(actions, ground_truth_list)
    print(f"Task Score: {final_score:.2f}")
    return final_score

async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set. Using dummy logic.")
    
    results = {}
    for task in TASKS:
        score = await run_task(task)
        results[task.name] = score
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    for name, score in results.items():
        print(f"{name}: {score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
