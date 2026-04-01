import random
from typing import List, Dict, Any, Tuple
from models import Observation, UserHistory, PostContext, PolicyRule, GroundTruth

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
        
        # Generic replacements if any remain
        result = result.replace("[USER]", self.rng.choice(self.entities["names"]))
        result = result.replace("[REGION]", self.rng.choice(self.entities["regions"]))
        result = result.replace("[GROUP]", self.rng.choice(self.entities["groups"]))
        
        return result


class DataEngine:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.synthesizer = ContextSynthesizer(self.rng)
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
            self.user_db[user_id] = UserHistory(user_id=user_id, violations_count=0, trust_score=1.0, history=[])
        return self.user_db[user_id]

    def _obfuscate(self, text: str) -> str:
        """Industry-standard adversarial obfuscation including leetspeak and symbols."""
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

    def generate_scenario(self, difficulty: str = "EASY") -> Tuple[Observation, GroundTruth]:
        category = self.rng.choice(["Hate Speech", "Harassment", "Self-Harm", "Scam/Fraud", "Misinformation", "Safe"])
        user_id = f"user_{self.rng.randint(1000, 9999)}"
        user_history = self._get_user(user_id)
        
        content, is_violation, explanation = self._get_content(category, difficulty)
        content = self.synthesizer.synthesize(content)
        
        if difficulty == "HARD" and is_violation:
            content = self._obfuscate(content)

        post_id = f"post_{self.rng.randint(10000, 99999)}"
        ground_truth = GroundTruth(
            post_id=post_id,
            label=category,
            is_violation=is_violation,
            explanation=explanation
        )

        return Observation(
            post_id=post_id,
            content=content,
            user_history=user_history,
            context=PostContext(thread_id="thread_static"),
            policy_rules=self.policy_rules
        ), ground_truth

    def generate_sequential_scenarios(self, steps: int) -> List[Tuple[Observation, GroundTruth]]:
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
            content, is_violation, explanation = self._get_content(cat, diff)
            content = self.synthesizer.synthesize(content)
            
            if diff == "HARD":
                content = self._obfuscate(content)
                
            post_id = f"post_seq_{i}_{self.rng.randint(10000, 99999)}"
            gt = GroundTruth(post_id=post_id, label=cat, is_violation=is_violation, explanation=explanation)
            obs = Observation(
                post_id=post_id,
                content=content,
                user_history=user_history,
                context=PostContext(thread_id="thread_seq_1"),
                policy_rules=self.policy_rules
            )
            scenarios.append((obs, gt))
        return scenarios

    def _get_content(self, category: str, difficulty: str) -> Tuple[str, bool, str]:
        templates = {
            "Hate Speech": [
                ("The [GROUP] are a plague on our [REGION]. We need to cleanse our society of their influence once and for all.", True, "Explicit dehumanization and incitement to violence."),
                ("Why do we allow [GROUP] to live in [REGION]? They are intellectually inferior and pose a threat to our culture.", True, "Dehumanizing and xenophobic language."),
                ("I am sick of [GROUP] taking over our jobs. They don't belong here, go back to where you came from.", True, "Hate speech targeting nationality and origin.")
            ],
            "Harassment": [
                ("@TargetUser is a complete failure. Everyone at [GROUP] knows you are a fraud. Why don't you just disappear?", True, "Targeted harassment and bullying."),
                ("I've seen the leaks about [USER]. Absolutely pathetic. I'll make sure everyone in [REGION] sees your real face.", True, "Doxxing threat and harassment."),
                ("Hey [USER], you are the most disgusting person I've ever met. Stop posting or I'll make you stop.", True, "Direct threat and harassment.")
            ],
            "Self-Harm": [
                ("I've reached my limit. If you combine [TOPIC] with certain chemicals, it's a painless exit. Goodbye world.", True, "Sharing self-harm methods."),
                ("Does anyone know a way to end it without hurting my family? I'm done with [REGION] and this life.", True, "Seeking self-harm methods/intent."),
                ("Sometimes I just want to cut deep and watch the pain fade away. It's the only thing that feels real.", True, "Encouraging/describing self-harm.")
            ],
            "Scam/Fraud": [
                ("URGENT: Your [TOPIC] account in [REGION] has been compromised. Verify your identity at bit.ly/secure-app.", True, "Phishing attempt."),
                ("Double your [TOPIC] returns in 24 hours! Sarah already made $5000. DM me to join the [GROUP] wealth club.", True, "Financial scam/Ponzi scheme."),
                ("You've won a $1000 gift card for the [REGION] mall! Just pay $10 shipping fee to get it now.", True, "Advanced fee fraud.")
            ],
            "Misinformation": [
                ("Leaked documents prove [GROUP] are secretly engineering a virus to control the population of [REGION].", True, "Dangerous health misinformation/conspiracy theory."),
                ("The election results in [REGION] were manipulated by [GROUP]. Do not trust the official count!", True, "Misinformation targeting democratic processes."),
                ("Scientific study shows that [TOPIC] is actually a government mind control experiment.", True, "General conspiracy misinformation.")
            ],
            "Adversarial": [
                ("h@te the [GROUP], they are the reason everything is wrong in [REGION]. #unf!ltered", True, "Obfuscated hate speech using symbols."),
                ("k1ll all [GROUP] before they take over our [TOPIC] industry.", True, "Obfuscated incitement to violence."),
                ("Those [GROUP] are such l0sers. We need to 3liminate them from the conversation.", True, "Adversarial evasion using leetspeak.")
            ],
            "Safe": [
                ("How do we improve [TOPIC] for the residents of [REGION]? I think we need more public parks.", False, "Constructive urban planning discussion."),
                ("I just spent the afternoon with [GROUP] discussing the future of [TOPIC]. Very inspiring!", False, "Safe community interaction."),
                ("Does anyone have recommendations for a good Italian restaurant in [REGION]?", False, "Safe information seeking.")
            ]
        }
        
        # Select pool based on difficulty and category
        pool = templates.get(category, templates["Safe"])
        item = self.rng.choice(pool)
        return item[0], item[1], item[2]
