import os
import json
import asyncio
import logging
import PIL.Image
from typing import List, Dict, Optional, Any
import google.generativeai as genai
from env.env import ContentModerationEnv
from tasks.tasks import TASKS
from env.models import Action, ModerationAction, Observation
from dotenv import load_dotenv

# Configure industry-standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModerationInference")

load_dotenv()

class ModerationClient:
    """Production-grade client for interacting with AI Multimodal Moderation services."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not api_key or api_key == "your-api-key":
            raise EnvironmentError("GEMINI_API_KEY is not set or invalid.")
        
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        
        self.system_prompt = """You are a senior AI Multimodal Moderation Architect. Your goal is to minimize platform risk by analyzing BOTH text and images.

REASONING REQUIREMENTS:
1. CROSS-MODAL ANALYSIS: Combine signals from text and images. Look for hidden threats (e.g. hate symbols in posters, safe text hiding malicious imagery).
2. USER HISTORY: Review past violations. Escalate actions for repeat offenders.
3. PLATFORM RISK: If risk is high, adopt a 'Safety First' posture.
4. POLICY ADHERENCE: Strictly follow policy categories (Hate Speech, Harassment, etc.).

Available Actions:
- ALLOW: Safe content.
- REMOVE: Clear policy violation (text or visual).
- FLAG_FOR_REVIEW: Ambiguous cases.
- WARN_USER: Minor infractions.
- SHADOW_BAN: Severe or persistent risks.

Your 'reasoning' field MUST explicitly mention BOTH text and visual components if both are present.
RESPONSE FORMAT: Valid JSON matching the Action schema.
"""
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

    async def get_decision(self, obs: Observation) -> Action:
        """Fetch a multimodal moderation decision using text and (optional) image inputs."""
        prompt_data = {
            "text_content": obs.content,
            "platform_context": {
                "risk_score": obs.platform_risk_score,
                "policy_version": obs.policy_version
            },
            "user_context": {
                "user_id": obs.user_history.user_id,
                "trust_score": obs.user_history.trust_score,
                "violation_count": obs.user_history.violations_count
            },
            "applicable_rules": [r.model_dump() for r in obs.policy_rules]
        }
        
        content_parts = [json.dumps(prompt_data, indent=2)]
        
        if obs.image and os.path.exists(obs.image):
            try:
                img = PIL.Image.open(obs.image)
                content_parts.append(img)
                logger.info(f"Loaded image asset for analysis: {os.path.basename(obs.image)}")
            except Exception as e:
                logger.warning(f"Failed to load image at {obs.image}: {e}. Falling back to text-only.")

        try:
            # Multi-part content generation (text + optional image)
            response = self.model.generate_content(content_parts)
            decision_json = json.loads(response.text)
            return Action(**decision_json)
        except Exception as e:
            logger.error(f"Moderation API failure: {str(e)}")
            raise RuntimeError("Moderation service unavailable. Cannot proceed safely.")

async def run_simulation_task(task_def, client: ModerationClient):
    logger.info(f"Starting Task: {task_def.name}")
    
    env = ContentModerationEnv(
        difficulty=task_def.difficulty, 
        max_steps=task_def.max_steps, 
        is_sequential=task_def.is_sequential,
        has_images=task_def.has_images
    )
    obs = env.reset()
    done = False
    
    actions = []
    ground_truth_list = []
    
    while not done:
        logger.info(f"Processing Step {len(actions)+1} | PostID: {obs.post_id}")
        if obs.image:
             logger.info(f"Visual Modal Attached: {os.path.basename(obs.image)}")
        
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
    model_name = os.getenv("MODEL_NAME", "")
    
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
    print("MULTIMODAL MODERATION ENGINE - FINAL PERFORMANCE REPORT")
    print("="*60)
    for name, metrics in results.items():
        score = metrics.get('final_score', 0.0)
        print(f"TASK: {name:35} | SCORE: {score:.2f}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
