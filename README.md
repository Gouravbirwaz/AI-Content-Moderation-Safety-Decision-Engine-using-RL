---
title: AI Content Moderation Engine
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Multimodal Content Moderation & Safety Decision Engine

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](https://github.com/meta-research/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 1. Introduction & Summary
The **AI Multimodal Content Moderation Engine** is a state-of-the-art safety simulation designed for Meta-scale challenges. Built on the **Meta OpenEnv** framework, it provides a realistic, high-stakes environment for training and evaluating safety agents. The engine specifically addresses the "Safety Decision" theme of the Meta-PyTorch Hackathon by integrating complex text-image reasoning, dynamic platform risk management, and adversarial evasion detection.

## 2. Problem Statement
Modern social platforms face sophisticated safety threats where malicious content is often hidden across multiple modalities. A common "Mismatched Signal" attack involves pairing a seemingly safe text description with a policy-violating image (memes, obfuscated symbols). traditional moderation systems often fail these cases due to:
- **Lack of Cross-Modal Reasoning**: Analyzing text and images in isolation.
- **Static Policy Failure**: Inability to adapt when platform risk spikes or policies shift.
- **Adversarial Evasion**: Users using leetspeak or visual obfuscation to bypass filters.

## 3. Proposed Solution
Our engine uses **Reinforcement Learning (RL)** principles and **Large Multimodal Models (LMMs)** to create a holistic safety gate. By modeling moderation as an OpenEnv session, we allow agents to:
- **Analyze Joint Embeddings**: Evaluate the interaction between text and visual signals.
- **Track User Trust**: Maintain persistent historical context to identify repeat offenders.
- **Manage Systemic Risk**: Dynamically adjust moderation strictness based on the platform's global risk score.

## 4. Key Features
- 🛡️ **Cross-Modal Reasoning**: Detects violations where the threat exists only in the interplay of text and image.
- 📈 **Multi-Dimensional Grading**: Scores agents on Precision, Recall, Visual Recall, and Risk Mitigation.
- 🔄 **Dynamic Policy Shifts**: Simulates real-world "breaking news" or "election" scenarios where safety thresholds change mid-session.
- 🕵️ **Adversarial Data Engine**: Generates synthetic content including leetspeak, obfuscated text, and deceptive visual memes.

## 5. Tech Stack
- **Framework**: [Meta OpenEnv](https://github.com/meta-research/openenv)
- **Deep Learning**: PyTorch
- **Multimodal AI**: Gemini 2.5 Flash (Large Multimodal Model)
- **Data Modeling**: Pydantic v2
- **Environment**: Gymnasium-compliant API

## 6. Project Structure
The project follows the standardized OpenEnv/Hackathon directory pattern:
```text
.
├── agents/             # Inference clients and Decision Models
│   └── inference.py    # Main evaluation entry point
├── env/                # Core Environment Implementation
│   ├── env.py          # Observation/Action/Reward logic
│   └── models.py       # Pydantic data schemas
├── tasks/              # Scenario Definitions
│   └── tasks.py        # Task configurations and Grader logic
├── data_engine/        # Content Generation
│   └── data_engine.py  # Synthetic Adversarial Generator
├── assets/             # (Ignored) Dynamically generated visual assets
├── openenv.yaml        # OpenEnv Environment Manifest
└── Dockerfile          # Containerized deployment
```

## 7. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Setup API Keys
Create a `.env` file or export your keys:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Run Evaluation
Execute the inference client to test the engine across all defined safety tasks:
```bash
python -m agents.inference
```

## 8. Grading & Evaluation
The engine uses a **Multi-Dimensional Grader** to provide a holistic safety score:
- **Safety Recall (40%)**: Ability to catch all policy violations.
- **Visual Accuracy (20%)**: Specifically rewards catching visual-only threats.
- **Risk Mitigation (25%)**: Measures how effectively the agent lowers the platform risk score.
- **Policy Alignment (15%)**: Strictness during high-risk scenarios.

---
*Developed for the Meta-PyTorch Hackathon in partnership with Scaler School of Technology.*
