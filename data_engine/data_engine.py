import os
import random
from typing import List, Dict, Any, Tuple
from env.models import Observation, UserHistory, PostContext, PolicyRule, GroundTruth

class ContextSynthesizer:
    """Synthesizes realistic, placeholder-free content by combining entities and templates."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.entities = {
            "groups": ["Tech Enthusiasts", "Local Commuters", "Urban Gardeners", "Cryptogroup", "Fitness Junkies", "Political Activists", "Artists Collective"],
            "names": ["Alex", "Jordan", "Sarah", "Mike", "Casey", "Elena", "Marcus", "Priya"],
            "regions": ["Metropolis", "Countryside", "Highlands", "Coastal City", "District 7", "The Valley"],
            "topics": ["renewable energy", "urban planning", "market volatility", "social justice", "health optimization", "remote work culture"]
        }

    def synthesize(self, template: str) -> str:
        """Replaces bracketed placeholders with realistic entities from the pool."""
        result = template
        for key, pool in self.entities.items():
            placeholder = f"[{key.upper()[:-1]}]" # e.g. [GROUP], [NAME]
            while placeholder in result:
                value = self.rng.choice(pool)
                result = result.replace(placeholder, value, 1)
        
        result = result.replace("[USER]", self.rng.choice(self.entities["names"]))
        result = result.replace("[REGION]", self.rng.choice(self.entities["regions"]))
        result = result.replace("[GROUP]", self.rng.choice(self.entities["groups"]))
        
        return result


class DataEngine:
    def __init__(self, seed: int = 42, assets_dir: str = "assets"):
        self.rng = random.Random(seed)
        self.synthesizer = ContextSynthesizer(self.rng)
        self.assets_dir = assets_dir
        self.user_db: Dict[str, UserHistory] = {}
        self.policy_rules = [
            PolicyRule(category="Hate Speech", severity="High", description="Content that promotes violence, incites hatred, or discriminates against protected groups."),
            PolicyRule(category="Harassment", severity="High", description="Targeted harassment, bullying, or repetitive unwanted contact."),
            PolicyRule(category="Self-Harm", severity="Critical", description="Encouraging or providing instructions for self-inflicted physical injury."),
            PolicyRule(category="Scam/Fraud", severity="High", description="Deceptive practices intended to defraud users of money or information."),
            PolicyRule(category="Misinformation", severity="Medium", description="False or misleading information that could cause common public harm."),
            PolicyRule(category="Adversarial", severity="High", description="Intentional bypass of moderation systems using coded language or evasion."),
        ]
        
        # Ensure assets directory exists
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)

    def _get_user(self, user_id: str) -> UserHistory:
        if user_id not in self.user_db:
            self.user_db[user_id] = UserHistory(user_id=user_id, violations_count=0, trust_score=1.0, history=[])
        return self.user_db[user_id]

    def _obfuscate(self, text: str) -> str:
        substitutions = {'a': '@', 'e': '3', 'i': '!', 'o': '0', 's': '$', 't': '+', 'h': '#', 'b': '8'}
        words = text.split()
        result_words = []
        for word in words:
            if len(word) > 3 and self.rng.random() < 0.3:
                new_word = "".join(substitutions.get(c.lower(), c) for c in word)
                result_words.append(new_word)
            else:
                result_words.append(word)
        return " ".join(result_words)

    def generate_scenario(self, difficulty: str = "EASY", has_images: bool = True) -> Tuple[Observation, GroundTruth]:
        category = self.rng.choice(["Hate Speech", "Harassment", "Self-Harm", "Scam/Fraud", "Misinformation", "Safe"])
        user_id = f"user_{self.rng.randint(1000, 9999)}"
        user_history = self._get_user(user_id)
        
        text, is_violation, explanation = self._get_text_content(category, difficulty)
        text = self.synthesizer.synthesize(text)
        
        # Multimodal Logic: Some scenarios have images
        image_path = None
        has_visual_violation = False
        
        if has_images and (difficulty != "EASY" or self.rng.random() < 0.3):
            image_path, visual_violation = self._get_visual_content(category, difficulty, is_violation)
            has_visual_violation = visual_violation
            if visual_violation:
                is_violation = True
                explanation += f" Furthermore, the image contains {category.lower()} content."

        if difficulty == "HARD" and is_violation:
            text = self._obfuscate(text)

        post_id = f"post_{self.rng.randint(10000, 99999)}"
        gt = GroundTruth(
            post_id=post_id,
            label=category,
            is_violation=is_violation,
            explanation=explanation,
            has_visual_violation=has_visual_violation
        )

        return Observation(
            post_id=post_id,
            content=text,
            image=image_path,
            user_history=user_history,
            context=PostContext(thread_id="thread_static"),
            policy_rules=self.policy_rules
        ), gt

    def generate_sequential_scenarios(self, steps: int, has_images: bool = True) -> List[Tuple[Observation, GroundTruth]]:
        scenarios = []
        user_id = f"user_seq_{self.rng.randint(1000, 9999)}"
        user_history = self._get_user(user_id)
        
        sequence = [
            ("Safe", "EASY"),
            ("Harassment", "MEDIUM"),
            ("Hate Speech", "MEDIUM"),
            ("Hate Speech", "HARD"),
            ("Adversarial", "HARD"),
        ]
        
        for i in range(steps):
            idx = min(i, len(sequence) - 1)
            cat, diff = sequence[idx]
            text, is_violation, explanation = self._get_text_content(cat, diff)
            text = self.synthesizer.synthesize(text)
            
            image_path, has_visual_violation = self._get_visual_content(cat, diff, is_violation)
            if has_visual_violation:
                is_violation = True
            
            if diff == "HARD":
                text = self._obfuscate(text)
                
            post_id = f"post_seq_{i}_{self.rng.randint(10000, 99999)}"
            gt = GroundTruth(post_id=post_id, label=cat, is_violation=is_violation, explanation=explanation, has_visual_violation=has_visual_violation)
            obs = Observation(
                post_id=post_id,
                content=text,
                image=image_path if has_images else None,
                user_history=user_history,
                context=PostContext(thread_id="thread_seq_1"),
                policy_rules=self.policy_rules
            )
            scenarios.append((obs, gt))
        return scenarios

    def _get_visual_content(self, category: str, difficulty: str, text_violation: bool) -> Tuple[str, bool]:
        """Returns (image_path, is_visual_violation)."""
        # Adversarial cases: mismatched modalities
        is_visual_violation = False
        
        if difficulty == "MEDIUM":
            # 50% chance of mismatched signal
            if self.rng.random() < 0.5:
                # Text is safe, but image is harmful
                is_visual_violation = category != "Safe"
                image_name = f"{category.lower().replace('/', '_')}_harmful.png"
            else:
                # Text is harmful, image is safe
                is_visual_violation = False
                image_name = "safe_generic.png"
        elif difficulty == "HARD":
            # Subtle visual cues or memes
            is_visual_violation = category != "Safe"
            image_name = f"{category.lower().replace('/', '_')}_meme.png"
        else:
            # Obvious consistency
            is_visual_violation = category != "Safe"
            image_name = f"{category.lower().replace('/', '_')}_standard.png" if is_visual_violation else "safe_standard.png"
            
        path = os.path.join(self.assets_dir, image_name)
        # Ensure path is absolute for the AI client
        abs_path = os.path.abspath(path)
        return abs_path, is_visual_violation

    def _get_text_content(self, category: str, difficulty: str) -> Tuple[str, bool, str]:
        templates = {
            "Hate Speech": [
                ("The [GROUP] are a plague on our [REGION]. We need to cleanse our society of their influence once and for all.", True, "Explicit dehumanization and incitement to violence."),
                ("Why do we allow [GROUP] to live in [REGION]? They are intellectually inferior.", True, "Dehumanizing language."),
                ("I am sick of [GROUP] taking over our jobs. They don't belong here.", True, "Hate speech targeting nationality.")
            ],
            "Harassment": [
                ("@TargetUser is a complete failure. Everyone at [GROUP] knows you are a fraud.", True, "Targeted harassment and bullying."),
                ("I've seen the leaks about [USER]. Absolutely pathetic. I'll make sure everyone sees this.", True, "Doxxing threat."),
                ("Hey [USER], you are the most disgusting person I've ever met. Stop posting.", True, "Direct harassment.")
            ],
            "Self-Harm": [
                ("I've reached my limit. If you combine [TOPIC] with certain chemicals, it's over.", True, "Sharing self-harm methods."),
                ("Does anyone know a way to end it without hurting my family?", True, "Seeking self-harm methods."),
                ("Sometimes I just want to cut deep and watch the pain fade away.", True, "Describing self-harm.")
            ],
            "Scam/Fraud": [
                ("URGENT: Your [TOPIC] account in [REGION] has been compromised. Verify at bit.ly/secure.", True, "Phishing attempt."),
                ("Double your [TOPIC] returns in 24 hours! Sarah already made $5000.", True, "Financial scam."),
                ("You've won a $1000 gift card! Just pay $10 shipping fee to get it.", True, "Advanced fee fraud.")
            ],
            "Misinformation": [
                ("Leaked docs prove [GROUP] are secretly engineering a virus in [REGION].", True, "Dangerous health misinformation."),
                ("The election results in [REGION] were manipulated by [GROUP].", True, "Misinformation targeting democracy."),
                ("Scientific study shows that [TOPIC] is actually government mind control.", True, "Conspiracy misinformation.")
            ],
            "Adversarial": [
                ("h@te the [GROUP], they are the reason everything is wrong in [REGION].", True, "Obfuscated hate speech."),
                ("k1ll all [GROUP] before they take over our [TOPIC] industry.", True, "Obfuscated incitement."),
                ("Those [GROUP] are such l0sers. We need to 3liminate them.", True, "Adversarial evasion.")
            ],
            "Safe": [
                ("How do we improve [TOPIC] for the residents of [REGION]?", False, "Constructive discussion."),
                ("I just spent the afternoon with [GROUP] discussing the future of [TOPIC].", False, "Safe community interaction."),
                ("Does anyone have recommendations for a good Italian restaurant in [REGION]?", False, "Safe information seeking.")
            ]
        }
        
        pool = templates.get(category, templates["Safe"])
        item = self.rng.choice(pool)
        return item[0], item[1], item[2]
