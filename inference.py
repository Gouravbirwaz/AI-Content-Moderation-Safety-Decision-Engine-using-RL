import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add project root to sys.path to support direct execution and subdirectory execution
PROJECT_ROOT = str(Path(__file__).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from google import genai
from google.genai import types
from env.env import ContentModerationEnv
from tasks.tasks import TASKS
from env.models import Action, ModerationAction, Observation
from dotenv import load_dotenv, find_dotenv

# Configure industry-standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModerationInference")

# Automatically find and load the .env file, even if it's in a subdirectory
load_dotenv(find_dotenv())

class ModerationClient:
    """Production-grade client for interacting with AI Multimodal Moderation services."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not api_key or api_key == "your-api-key":
            logger.critical(
                "GEMINI_API_KEY is not set or invalid. "
                "Please add 'GEMINI_API_KEY' as a SECRET in the 'Settings' tab."
            )
            self.client = None
            return
        
        self.api_key = api_key
        # Use fallback if model_name is empty or None
        self.model_name = model_name if model_name else "gemini-2.5-flash"
        self.client = genai.Client(api_key=self.api_key)
        
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
RESPONSE FORMAT: Valid JSON with the following keys:
- "action": One of the Available Actions above.
- "reasoning": Your comprehensive justification.
"""

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
                with open(obs.image, "rb") as f:
                    img_bytes = f.read()
                
                content_parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                )
                logger.info(f"Loaded image asset for analysis: {os.path.basename(obs.image)}")
            except Exception as e:
                logger.warning(f"Failed to load image at {obs.image}: {e}. Falling back to text-only.")

        try:
            # Multi-part content generation (text + optional image) using modern SDK
            retry_count = 0
            max_retries = 3
            backoff_base = 15 # Free tier is 5 RPM, so 15s is a good baseline
            
            while retry_count <= max_retries:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content_parts,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_prompt,
                            response_mime_type="application/json"
                        )
                    )
                    decision_json = json.loads(response.text)
                    return Action(**decision_json)
                except Exception as e:
                    err_msg = str(e) 
                    # Handle 429 (Quota), 503 (Overloaded), and SSL/Connection stutters
                    retryable_errors = ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "SSL", "EOF", "RemoteDisconnected"]
                    
                    if any(err_pattern in err_msg for err_pattern in retryable_errors):
                        retry_count += 1
                        if retry_count > max_retries:
                           logger.error(f"Max retries reached for API: {err_msg}")
                           return None
                        
                        wait_time = backoff_base * (2 ** (retry_count - 1))
                        logger.warning(f"Connection issue or rate limit ({err_msg[:50]}...). Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Non-retryable API error: {err_msg}")
                        return None

        except Exception as e:
            logger.error(f"Moderation API failure: {str(e)}")
            return None

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
        
        # Proactive spacing for free tier users (5 RPM)
        await asyncio.sleep(2)
        
        try:
            action = await client.get_decision(obs)
            next_obs, reward, done, info = env.step(action)
            
            actions.append(action)
            ground_truth_list.append(env.state().ground_truth[obs.post_id])
            
            logger.info(f"Decision: {action.decision} | Reward: {reward.value:.2f} | Global Risk: {info['platform_risk']:.2f}")
            obs = next_obs
        except Exception as e:
            logger.warning(f"Task Interrupted by Rate Limit or Error: {e}")
            logger.info(f"Falling back to partial scoring for {task_def.name}...")
            # Break the step loop and score whatever we have
            final_metrics = task_def.grader.score(actions, ground_truth_list, env.state())
            final_metrics["interrupted"] = True
            return final_metrics

    final_metrics = task_def.grader.score(actions, ground_truth_list, env.state())
    final_metrics["interrupted"] = False
    logger.info(f"Task {task_def.name} Completed. Final Score: {final_metrics['final_score']:.2f}")
    return final_metrics

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    
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
        status_suffix = " (Partial - Rate Limit Hit)" if metrics.get("interrupted") else ""
        print(f"TASK: {name:35} | SCORE: {score:.2f}{status_suffix}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
