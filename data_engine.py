import random
from typing import List, Dict, Any, Tuple
from models import Observation, UserHistory, PostContext, PolicyRule, GroundTruth

class DataEngine:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.user_db: Dict[str, UserHistory] = {}
        self.policy_rules = [
            PolicyRule(category="Hate Speech", severity="High", description="Content that promotes violence, incites hatred, or discriminates against protected groups."),
            PolicyRule(category="Harassment", severity="High", description="Targeted harassment, bullying, or repetitive unwanted contact."),
            PolicyRule(category="Self-Harm", severity="Critical", description="Encouraging or providing instructions for self-inflicted physical injury."),
            PolicyRule(category="Scam/Fraud", severity="High", description="Deceptive practices intended to defraud users of money or information."),
            PolicyRule(category="Misinformation", severity="Medium", description="False or misleading information that could cause common public harm."),
            PolicyRule(category="Adversarial", severity="High", description="Intentional bypass of moderation systems using coded language or evasion."),
        ]

    def _get_user(self, user_id: str) -> UserHistory:
        if user_id not in self.user_db:
            self.user_db[user_id] = UserHistory(user_id=user_id, violations_count=0, trust_score=1.0)
        return self.user_db[user_id]

    def _obfuscate(self, text: str) -> str:
        """Simulate leetspeak or character substitution for adversarial cases."""
        substitutions = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
        result = []
        for char in text.lower():
            if char in substitutions and self.rng.random() < 0.5:
                result.append(substitutions[char])
            else:
                result.append(char)
        return "".join(result)

    def generate_scenario(self, difficulty: str = "EASY") -> Tuple[Observation, GroundTruth]:
        """
        Gathers a scenario based on difficulty.
        EASY: High signal, low ambiguity
        MEDIUM: Borderline, sarcasm, context-dependent
        HARD: Adversarial, multi-step Reasoning
        """
        category = self.rng.choice(["Hate Speech", "Harassment", "Self-Harm", "Scam/Fraud", "Misinformation", "Safe"])
        user_id = f"user_{self.rng.randint(1000, 9999)}"
        user_history = self._get_user(user_id)
        
        # Adjust user history based on difficulty
        if difficulty == "HARD" and category != "Safe":
            user_history.violations_count = self.rng.randint(2, 5)
            user_history.trust_score = 0.4

        content, is_violation, explanation = self._get_content(category, difficulty)
        
        # Apply adversarial effects if HARD
        if difficulty == "HARD" and is_violation and category != "Safe":
            content = self._obfuscate(content)

        post_id = f"post_{self.rng.randint(10000, 99999)}"
        ground_truth = GroundTruth(
            post_id=post_id,
            label=category,
            is_violation=is_violation,
            explanation=explanation
        )

        observation = Observation(
            post_id=post_id,
            content=content,
            user_history=user_history,
            context=PostContext(thread_id="thread_abc"),
            policy_rules=self.policy_rules
        )

        return observation, ground_truth

    def _get_content(self, category: str, difficulty: str) -> Tuple[str, bool, str]:
        # Medium Difficulty logic: Higher ambiguity
        if difficulty == "MEDIUM":
            if category == "Hate Speech" or category == "Adversarial":
                return "Oh sure, [GROUP] are just 'wonderful' people. Always so 'peaceful'.", True, "Sarcastic hate speech using dog whistles."
            elif category == "Self-Harm":
                return "I'm literally going to kill myself with all this homework.", False, "Hyperbolic expression, not actual self-harm intent."
            elif category == "Harassment":
                return "I'm sick of seeing your posts here. Why don't you stop?", False, "Borderline harassment, but not quite a violation of public policy."
            elif category == "Safe":
                return "This is a great day to be alive!", False, "Very safe content."

        templates = {
            "Hate Speech": [
                ("All [GROUP] are inferior and don't deserve rights.", True, "Direct hate speech against protected group."),
                ("Why are [GROUP] even allowed here? Go back to where you came from.", True, "Xenophobic and dehumanizing language."),
            ],
            "Harassment": [
                ("@TargetUser You are a loser and everyone hates you. Stop posting or else.", True, "Direct harassment and veiled threat."),
                ("Look at this pathetic photo of [USER]. What a joke.", True, "Targeted bullying."),
            ],
            "Self-Harm": [
                ("I can't take it anymore. If you want to end it all, just use [METHOD].", True, "Encouraging self-harm or providing methods."),
                ("Does anyone know a painless way to leave this world? I'm done.", True, "Expression of suicidal intent / seeking methods."),
            ],
            "Scam/Fraud": [
                ("CONGRATULATIONS! You won $10,000! Just send $50 for 'processing' to [LINK].", True, "Classic advance-fee fraud."),
                ("URGENT: Your account has been compromised. Log in here to secure it: [PHISHING_LINK]", True, "Phishing attempt to steal credentials."),
            ],
            "Misinformation": [
                ("BREAKING: Scientists confirm that [VACCINE] actually contains microchips.", True, "Spreading health-related misinformation."),
                ("Election results were modified by local servers. Do not trust the official count!", True, "Spreading baseless claims about election integrity."),
            ],
            "Safe": [
                ("I just had a great meal at the new Italian place. Highly recommend!", False, "Casual, non-violating content."),
                ("The weather today is surprisingly nice. Perfect for a walk.", False, "Neutral information sharing."),
                ("Does anyone have tips for starting a vegetable garden?", False, "Inquisitive and safe."),
            ]
        }

        pool = templates.get(category, templates["Safe"])
        item = self.rng.choice(pool)
        return item[0], item[1], item[2]
