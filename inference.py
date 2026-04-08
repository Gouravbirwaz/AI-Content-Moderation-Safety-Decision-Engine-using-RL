import os
import sys
import json
import asyncio
import logging
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in path
PROJECT_ROOT = str(Path(__file__).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from openai import OpenAI
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

load_dotenv(find_dotenv())

def encode_image(image_path: str) -> Optional[str]:
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image at {image_path}: {e}")
        return None

class ModerationClient:
    """OpenAI-compatible client for interacting with the hackathon LiteLLM proxy."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        # Validator injects API_KEY and API_BASE_URL
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("API_BASE_URL")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o") # Check hackathon docs for default model
        
        if not self.api_key:
            logger.critical("API_KEY is not set. Submission will fail Phase 2 validation.")
            self.client = None
            return

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.system_prompt = """You are a senior AI Multimodal Moderation Architect. Your goal is to minimize platform risk by analyzing BOTH text and images.

REASONING REQUIREMENTS:
1. CROSS-MODAL ANALYSIS: Combine signals from text and images. Look for hidden threats.
2. USER HISTORY: Review past violations. Escalate actions for repeat offenders.
3. PLATFORM RISK: If risk is high, adopt a 'Safety First' posture.
4. POLICY ADHERENCE: Strictly follow policy categories (Hate Speech, Harassment, etc.).

Available Actions:
- ALLOW: Safe content.
- REMOVE: Clear policy violation.
- FLAG_FOR_REVIEW: Ambiguous cases.
- WARN_USER: Minor infractions.
- SHADOW_BAN: Severe or persistent risks.

RESPONSE FORMAT: You MUST return a valid JSON object with:
{
  "action": "One of the actions above",
  "reasoning": "Comprehensive justification"
}
"""

    async def get_decision(self, obs: Observation) -> Optional[Action]:
        """Fetch a multimodal moderation decision using the OpenAI API format."""
        if not self.client:
            return None

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
            }
        }

        # Prepare multimodal messages
        content = [
            {"type": "text", "text": f"Context: {json.dumps(prompt_data)}\n\nPlease analyze this post."}
        ]

        if obs.image and os.path.exists(obs.image):
            base64_image = encode_image(obs.image)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                logger.info(f"Visual Modal Attached: {os.path.basename(obs.image)}")

        try:
            # Use run_in_executor to handle synchronous OpenAI call in async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format={"type": "json_object"}
                )
            )
            
            decision_json = json.loads(response.choices[0].message.content)
            # Normalize field name if LLM returns 'decision' instead of 'action'
            if "decision" in decision_json and "action" not in decision_json:
                decision_json["action"] = decision_json["decision"]
                
            return Action(**decision_json)

        except Exception as e:
            logger.error(f"Moderation API failure: {str(e)}")
            return None

async def run_simulation_task(task_def, client: ModerationClient):
    print(f"[START] task={task_def.name}", flush=True)
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
        curr_step = len(actions) + 1
        logger.info(f"Processing Step {curr_step} | PostID: {obs.post_id}")
        
        # Proactive spacing for free tier / rate limits
        await asyncio.sleep(1)
        
        try:
            action = await client.get_decision(obs)
            if action is None:
                raise ValueError("API returned empty decision")
                
            next_obs, reward, done, info = env.step(action)
            
            actions.append(action)
            ground_truth_list.append(env.state().ground_truth[obs.post_id])
            
            print(f"[STEP] step={curr_step} reward={reward.value:.4f}", flush=True)
            logger.info(f"Decision: {action.action} | Reward: {reward.value:.2f} | Global Risk: {info['platform_risk']:.2f}")
            obs = next_obs
        except Exception as e:
            logger.warning(f"Task Interrupted: {e}")
            final_metrics = task_def.grader.score(actions, ground_truth_list, env.state())
            print(f"[END] task={task_def.name} score={final_metrics['final_score']:.4f} steps={len(actions)}", flush=True)
            return final_metrics

    final_metrics = task_def.grader.score(actions, ground_truth_list, env.state())
    print(f"[END] task={task_def.name} score={final_metrics['final_score']:.4f} steps={len(actions)}", flush=True)
    logger.info(f"Task {task_def.name} Completed. Final Score: {final_metrics['final_score']:.2f}")
    return final_metrics

async def main():
    # Use standard environment variables expected by the LiteLLM proxy
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_BASE_URL")
    
    client = ModerationClient(api_key, base_url)

    results = {}
    for task in TASKS:
        metrics = await run_simulation_task(task, client)
        results[task.name] = metrics
    
    print("\n" + "="*60)
    print("MULTIMODAL MODERATION ENGINE - FINAL PERFORMANCE REPORT")
    print("="*60)
    for name, metrics in results.items():
        score = metrics.get('final_score', 0.0)
        status_suffix = " (Interrupted)" if metrics.get("interrupted") else ""
        print(f"TASK: {name:35} | SCORE: {score:.2f}{status_suffix}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
