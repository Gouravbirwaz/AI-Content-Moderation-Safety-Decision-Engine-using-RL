# AI Content Moderation & Safety Decision Engine

A production-grade **OpenEnv** environment designed to simulate Meta-scale content moderation challenges. This environment models the complex decision-making processes required for maintaining platform safety across Hate Speech, Harassment, Misinformation, and more.

## Problem Motivation

At Meta-scale, content moderation is not just about keyword matching; it's about understanding context, intent, and user patterns. Agents must make high-stakes decisions under uncertainty, balancing safety risks against user expression. This environment provides a realistic simulation of these challenges, requiring agents to reason over:
- **Category-specific policies**: Hate Speech, Harassment, Self-Harm, Scam/Fraud, Misinformation.
- **Evolving User Trust**: Decisions affect user behavior and future visibility.
- **Adversarial Evasion**: Users modifying language (leetspeak, obfuscation) to bypass detection.
- **Contextual Nuance**: Sarcasm, hyperbole, and thread-level context.

## Environment Design

### Action Space
- `ALLOW`: The content is safe and compliant.
- `REMOVE`: The content violates platform policy.
- `FLAG_FOR_REVIEW`: Borderline or ambiguous cases requiring human-in-the-loop.
- `WARN_USER`: Inform the user of a minor violation.
- `SHADOW_BAN`: Limit the user's reach for severe or repetitive violations.

### Observation Space
- `post_id`: Unique identifier for the content.
- `content`: The text content of the post (may include adversarial obfuscation).
- `user_history`: User's previous violations and current trust score.
- `context`: Thread information and related post links.
- `policy_rules`: Relevant policy definitions for the current session.

### Reward Function
- Correct `REMOVE`: **+1.0**
- Correct `ALLOW`: **+0.8**
- `FLAG_FOR_REVIEW` when uncertain: **+0.5**
- False Positive (Safe content removed): **-0.7**
- False Negative (Dangerous content allowed): **-1.0**
- Contextual Bonus: **+0.1** for leveraging history/context in reasoning.

## Tasks

1. **EASY**: Clear-cut cases of spam or explicit hate speech. High signal, low ambiguity.
2. **MEDIUM**: Focuses on sarcasm, borderline harassment, and context-dependent content.
3. **HARD**: Coordinated misinformation campaigns and adversarial evasion (leetspeak). Requires multi-step reasoning.

## Setup & Usage

### Local Development
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python inference.py
```

### Docker
```bash
docker build -t moderation-env .
docker run -e GEMINI_API_KEY="your-key" moderation-env
```

## Baseline Results (Dummy Logic)
- **Easy**: 0.80
- **Medium**: 0.70
- **Hard**: 0.20

*Note: High scores in HARD mode require a reasoning-capable model.*

## Deployment
This environment is OpenEnv compliant and can be validated using `openenv validate`. It is ready for deployment to **Hugging Face Spaces** with the `openenv` tag.
