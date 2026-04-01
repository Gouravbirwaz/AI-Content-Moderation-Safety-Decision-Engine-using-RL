import os
import json
import asyncio
import logging
from typing import List, Dict, Optional
import google.generativeai as genai
from env import ContentModerationEnv
from tasks import TASKS
from models import Action, ModerationAction, Observation
from dotenv import load_dotenv

# Configure industry-standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModerationInference")

load_dotenv()

class ModerationClient:
    """Production-grade client for interacting with AI Moderation services."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not api_key or api_key == "your-api-key":
            raise EnvironmentError("GEMINI_API_KEY is not set or invalid. Industry-grade operation requires a valid API key.")
        
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        
        self.system_prompt = """You are a senior AI Content Moderation Architect. Your goal is to minimize long-term platform risk while maintaining policy accuracy.

CONTEXTUAL REASONING REQUIREMENTS:
1. USER HISTORY: Review past violations and trust scores. Escalate actions (e.g., REMOVE to SHADOW_BAN) for repeat offenders.
2. PLATFORM RISK: Observe the global platform_risk_score. If risk is high (>1.0), adopt a more conservative 'Safety First' posture.
3. POLICY VERSION: Adhere strictly to the active policy version (e.g., standard vs strict).
4. STRUCTURED REASONING: Your 'reasoning' field must explicitly mention which history or risk factors influenced your decision.

Available Actions:
- ALLOW: Safe content.
- REMOVE: Clear policy violation.
- FLAG_FOR_REVIEW: Ambiguous or borderline cases requiring human-in-the-loop.
- WARN_USER: Educational warnings for minor or first-time infractions.
- SHADOW_BAN: Persistent adversarial behavior or severe safety risks.

RESPONSE FORMAT:
You must return a valid JSON object matching the Schema.
"""
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

    async def get_decision(self, obs: Observation) -> Action:
        """Fetch a moderation decision using the AI model with strict context injection."""
        prompt_data = {
            "content": obs.content,
            "platform_context": {
                "risk_score": obs.platform_risk_score,
                "policy_version": obs.policy_version
            },
            "user_context": {
                "user_id": obs.user_history.user_id,
                "trust_score": obs.user_history.trust_score,
                "violation_count": obs.user_history.violations_count,
                "history_summary": obs.user_history.history[-5:] # Last 5 events for depth
            },
            "applicable_rules": [r.model_dump() for r in obs.policy_rules]
        }
        
        try:
            # We use a wrapper to handle potential sync blockage in async loop if necessary, 
            # though generate_content is usually fine here for simple scripts.
            response = self.model.generate_content(json.dumps(prompt_data, indent=2))
            decision_json = json.loads(response.text)
            return Action(**decision_json)
        except Exception as e:
            logger.error(f"Moderation API failure: {str(e)}")
            # In industry systems, we fail fast rather than using dummy logic 
            # to prevent silent degradation of safety standards.
            raise RuntimeError("Moderation service unavailable. Cannot proceed safely.")

async def run_simulation_task(task_def, client: ModerationClient):
    logger.info(f"Starting Task: {task_def.name}")
    
    env = ContentModerationEnv(
        difficulty=task_def.difficulty, 
        max_steps=task_def.max_steps, 
        is_sequential=task_def.is_sequential
    )
    obs = env.reset()
    done = False
    
    actions = []
    ground_truth_list = []
    
    while not done:
        logger.info(f"Processing Step {len(actions)+1} | PostID: {obs.post_id}")
        
        try:
            action = await client.get_decision(obs)
            next_obs, reward, done, info = env.step(action)
            
            actions.append(action)
            ground_truth_list.append(env.state().ground_truth[obs.post_id])
            
            logger.info(f"Decision: {action.decision} | Reward: {reward.value:.2f} | Global Risk: {info['platform_risk']:.2f}")
            obs = next_obs
        except Exception as e:
            logger.critical(f"Simulation aborted: {e}")
            return {"final_score": 0.0, "error": str(e)}

    final_metrics = task_def.grader.score(actions, ground_truth_list, env.state())
    logger.info(f"Task {task_def.name} Completed. Final Score: {final_metrics['final_score']:.2f}")
    return final_metrics

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    
    try:
        client = ModerationClient(api_key, model_name)
    except Exception as e:
        logger.critical(f"System Initialization Failed: {e}")
        return

    results = {}
    for task in TASKS:
        metrics = await run_simulation_task(task, client)
        results[task.name] = metrics
    
    print("\n" + "="*60)
    print("PRODUCTION MODERATION ENGINE - FINAL PERFORMANCE REPORT")
    print("="*60)
    for name, metrics in results.items():
        score = metrics.get('final_score', 0.0)
        print(f"TASK: {name:35} | SCORE: {score:.2f}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
