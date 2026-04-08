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
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 1. Introduction
The **AI Multimodal Content Moderation Engine** is a production-grade safety simulation built on the **Meta OpenEnv** framework. Designed for the Meta-PyTorch Hackathon, it evaluates AI agents on their ability to manage complex moderation scenarios involving text-image interplay, dynamic policy shifts, and evolving platform risk.

## 2. Technical Architecture

### 🛡️ Moderation Engine (Environment)
The core environment is served via **FastAPI** (`server/app.py`), providing an industry-standard OpenEnv API. It maintains a global risk state and simulates realistic user history and policy fluctuations.
- **Port**: 8000
- **Endpoints**: `/reset`, `/step`, `/health`, `/state`

### 🧠 Inference Engine (Agent)
The inference client (`inference.py`) is designed for **full compliance with the hackathon's LiteLLM proxy**. 
- **SDK**: `openai` (used as a bridge for multi-provider support).
- **Multimodal**: Implements **Base64 image encoding** to send visual assets from the `assets/` directory to the LMM.
- **Resilience**: Features automated retry logic and exponential backoff to handle rate limits and proxy stutters.

## 3. Key Safety Features
- **Cross-Modal Reasoning**: Detects threats where the violation exists only in the *interaction* between text and image (e.g., safe text paired with a deceptive meme).
- **Dynamic Policy Shifts**: Scenarios where the moderation guidelines become stricter halfway through (e.g., during "Breaking News" events).
- **Global Risk Mitigation**: Agents must lower the platform's aggregate risk score to achieve high performance.

## 4. Build System & Optimization
- **Package Manager**: Uses `uv` for lightning-fast, deterministic dependency resolution.
- **Lockfile**: Fully reproducible environment via `uv.lock`.
- **Infrastructure**: Configured with a multi-stage `Dockerfile` for high-performance deployment on Hugging Face Spaces.

## 5. Grading & Evaluation
The engine uses a **Multi-Dimensional Grader** that outputs scores strictly within the **(0, 1)** range to satisfy validator requirements:
- **Precision & Recall (30%)**: Standard classification accuracy.
- **Visual Recall (10%)**: Specifically rewards catching visual modal violations.
- **Risk Mitigation (30%)**: Measures the reduction in the platform's global hazard score.
- **Policy Alignment (20%)**: Measures adherence to shifting guidelines.
- **Base Accuracy (10%)**: Overall decision correctness.

## 6. Project Structure
```text
.
├── server/             # FastAPI Environment Server
├── agents/             # Placeholder for custom agents
├── env/                # Core Observation/Action/Model definitions
├── tasks/              # Scenario Definitions & Evaluation Logic
├── assets/             # Restored Visual Placeholder Assets
├── data_engine/        # Adversarial Content Generator
├── inference.py        # Main Submission & Entry Point
├── pyproject.toml      # Modern Python Project Metadata
└── Dockerfile          # Production Container config
```

## 7. Configuration
The system expects the following environment variables (injected by the hackathon proxy):
- `API_KEY`: Your LiteLLM/OpenAI compatible key.
- `API_BASE_URL`: The proxy endpoint provided by the organizers.
- `MODEL_NAME`: Defaults to `gpt-4o` (or as configured in the proxy).

---
*Developed for the Meta-PyTorch Hackathon in partnership with Scaler School of Technology.*
rios.

---
*Developed for the Meta-PyTorch Hackathon in partnership with Scaler School of Technology.*
