# TechSara Solutions - Automated Candidate Evaluation & Interview Analysis Platform

A sophisticated **end-to-end recruitment automation system** that intelligently processes candidate interview videos, performs AI-powered evaluations across multiple dimensions, and generates data-driven assessment reports for hiring teams.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Evaluation Criteria](#evaluation-criteria)
- [Output Formats](#output-formats)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This platform automates the complete candidate evaluation workflow for recruitment teams. It ingests raw interview videos from Google Drive, performs automated transcription and analysis, and delivers structured candidate assessments with eye-contact metrics and multi-dimensional performance scoring.

**Use Case**: Rapidly screen and rank job candidates using AI-powered video analysis, eliminating manual review bottlenecks while maintaining assessment rigor.

**Key Benefits**:
- ✅ Handles 100+ candidates per batch
- ✅ 6-dimensional evaluation framework (intro, mock interview, projects, niche knowledge, resume alignment, tech skills)
- ✅ Eye-contact & gaze tracking with computer vision
- ✅ Automated report generation in Google Docs & Excel
- ✅ Enterprise-grade error handling & resilience
- ✅ Containerized for cloud deployment (AWS, GCP, etc.)

---

## Key Features

### 🎥 Video Processing
- Automated video download from Google Drive folder structures
- OpenAI GPT-4o speech-to-text transcription with speaker diarization
- Chunked processing for videos exceeding API duration limits
- Exponential backoff retry logic for transient failures

### 🤖 AI-Powered Evaluation
- **6-Dimensional Assessment Framework**:
  1. Introduction Video - Self-presentation clarity & professionalism
  2. Mock Interview - Q&A relevance, depth, and confidence
  3. Project Scenarios - Technical problem-solving & ownership
  4. Domain Knowledge - Niche/specialized expertise alignment
  5. Resume Validation - Experience credibility & depth
  6. Technology Skills - Hands-on tool proficiency

- GPT-4 LLM-based evaluation with customizable scoring rubrics
- Automated PASS/FAIL classification (40% threshold)
- Scripted vs. authentic response detection

### 👁️ Computer Vision Analysis
- **Real-time Eye Tracking**: MediaPipe FaceMesh for 468-point facial landmarks
- **Gaze Direction Analysis**: ONNX-based gaze estimation model (3D eye vector)
- **Face Identification**: InsightFace embeddings for multi-angle video tracking
- **Metrics Generated**:
  - Eye contact frequency & duration
  - Gaze direction consistency
  - Facial expression patterns
  - Head position & movement
  - CSV export for analytics

### 📊 Report Generation
- **Per-Candidate**: Individual Google Docs with consolidated analysis
- **Slot-Level**: Aggregate analysis documents for all candidates in interview batch
- **Excel Sheets**: Structured scoring data with candidate rankings
- **Automated Sharing**: Email notifications to hiring team

### 🔄 Batch Processing
- Non-interactive automation via environment variable slot selection
- Configurable minimum score thresholds (deliverables, eye contact)
- Top-N candidate filtering for high-volume screening

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT: Google Drive                        │
│       Video folders organized by Slot → Candidate           │
└────────────────────┬────────────────────────────────────────┘
                     │
     ┌───────────────▼───────────────┐
     │  test.py: Transcription       │
     │  (GPT-4o Speech-to-Text       │
     │   with Diarization)           │
     └───────────────┬───────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ test2.py: LLM Evaluation       │
     │ (6 prompt-based assessments    │
     │  against transcripts)          │
     └───────────────┬────────────────┘
                     │
    ┌────────────────┴──────────────┐
    │                               │
┌───▼──────────────┐    ┌──────────▼──────────────┐
│ test3.py:        │    │ test6.py: Eye Tracking  │
│ Individual Docs  │    │ (Computer Vision)       │
└───┬──────────────┘    └──────────┬──────────────┘
    │                              │
    └────────────────┬─────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ test4.py: Slot Consolidation   │
     │ (All Deliverables Analysis)    │
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ test5.py: Excel Generation     │
     │ (Deliverables Analysis Sheets) │
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ test7.py: Results Aggregation  │
     │ test8.py: Final Delivery       │
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │      OUTPUT: Google Drive      │
     │  • Analysis Google Docs        │
     │  • Excel Reports               │
     │  • Eye Tracking Videos         │
     │  • Metrics CSV Files           │
     └────────────────────────────────┘
```

---

## Technology Stack

### Core Processing
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Speech-to-Text** | OpenAI GPT-4o | Audio transcription with speaker diarization |
| **LLM Evaluation** | OpenAI GPT-4 / GPT-4 Mini | Multi-dimensional candidate assessment |
| **Computer Vision** | MediaPipe FaceMesh | 468-point facial landmark detection |
| **Face Recognition** | InsightFace | Face embedding & identification |
| **Gaze Estimation** | ONNX Runtime | 3D eye vector computation |

### Cloud & APIs
| Service | Purpose |
|---------|---------|
| **Google Drive API** | Video ingestion & report delivery |
| **Google Docs API** | Automated document creation |
| **Google Sheets API** | Excel spreadsheet generation |
| **OpenAI API** | Transcription + LLM evaluation |

### Data Processing
| Library | Version | Use Case |
|---------|---------|----------|
| pandas | 2.2.2 | Data aggregation & CSV handling |
| numpy | 1.26.4 | Numerical computations |
| opencv-python | 4.10.0.84 | Video frame extraction |
| openpyxl | 3.1.5 | Excel file generation |
| onnxruntime | 1.20.1 | ONNX model inference |

### Infrastructure
| Tool | Purpose |
|------|---------|
| **Docker** | Containerized deployment |
| **Docker Compose** | Multi-stage orchestration |
| **Python 3.11** | Runtime environment |

---

## Prerequisites

### System Requirements
- **Python**: 3.10+ (3.11 recommended for Docker)
- **RAM**: 8GB+ (16GB recommended for video processing)
- **Storage**: 50GB+ free space (for video caching, temporary files)
- **OS**: Windows, macOS, or Linux
- **GPU** (Optional): CUDA 11.8+ for accelerated video processing

### API Keys & Credentials
1. **OpenAI API Key**
   - Create account at [platform.openai.com](https://platform.openai.com)
   - Generate API key with transcription & chat model access
   - Minimum quota: ~$50-100/month for typical batch processing

2. **Google OAuth Credentials**
   - Create Google Cloud project at [console.cloud.google.com](https://console.cloud.google.com)
   - Enable APIs:
     - Google Drive API
     - Google Docs API
     - Google Sheets API
   - Create OAuth 2.0 client (Desktop Application)
   - Download `credentials.json` (store in project root)

3. **Google Drive Folder Structure**
   - Root folder: **2026/** (in Google Drive)
   - Pre-create **Candidate Result/** folder for outputs
   - Ensure team has access to shared folders

---

## Installation & Setup

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Transcript
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Credentials
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API credentials
# Required:
# - OPENAI_API_KEY=sk-proj-...
# - Google credentials.json in project root
```

### Step 5: Initialize Google OAuth
```bash
python test.py
# On first run, browser window opens for OAuth consent
# Generates token.json (auto-refresh handled)
```

### Step 6: Verify Installation
```bash
python -c "import openai, google.auth; print('✓ All dependencies loaded')"
```

---

## Configuration

### Environment Variables (.env)

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-key-here          # Required: API key for transcription & LLM
OPENAI_MODEL=gpt-4-turbo                       # LLM for evaluation (gpt-4-turbo, gpt-4o, etc.)
OPENAI_MODEL_MINI=gpt-4-turbo-mini             # Lightweight model for quick tasks

# Processing Configuration
SLOT_CHOICE=1                                   # Auto-select interview slot (1-based index)
                                                # Leave commented for interactive selection
MIN_DELIVERABLES=55                            # Minimum score % to pass deliverables (0-100)
MIN_EYE=70                                     # Minimum eye contact score % to pass (0-100)
TOP_N=5                                        # Return top N candidates in rankings

# Google Drive Configuration
USE_SHARED_DRIVES=false                        # Set true if using Google Team Drives
                                                # (Requires additional scopes)

# Logging & Debug (Optional)
LOG_LEVEL=INFO                                 # DEBUG, INFO, WARNING, ERROR
DEBUG_VIDEO_FRAMES=false                       # Save debug frames during eye tracking
CACHE_TRANSCRIPTS=true                         # Reuse cached .txt transcripts
```

### Evaluation Prompts

Edit prompt files to customize scoring rubrics:

```
├── intro-prompt.txt              # Introduction video criteria
├── mock-prompt.txt               # Mock interview Q&A rubric
├── project-scenario.txt          # Project explanation framework
├── niche-prompt.txt              # Domain expertise evaluation
├── CV-prompt.txt                 # Resume alignment assessment
└── Tools-Technology-prompt.txt   # Tech stack knowledge testing
```

Each prompt file contains structured scoring guidelines for GPT-4 LLM.

---

## Usage Guide

### Mode 1: Interactive (Manual Slot Selection)
```bash
# Runs full pipeline with user prompts for slot selection
python test.py              # Download & transcribe videos
python test2.py             # Evaluate transcripts
python test3.py             # Create individual analysis docs
python test4.py             # Create slot-level consolidated docs
python test5.py             # Generate Excel sheets
python test6.py             # Perform eye tracking analysis
python test7.py             # Aggregate results
python test8.py             # Final delivery package
```

### Mode 2: Batch Automation (SLOT_CHOICE in .env)
```bash
# Set SLOT_CHOICE=1 in .env, then run:
python test.py && python test2.py && python test3.py && \
python test4.py && python test5.py && python test6.py && \
python test7.py && python test8.py
```

### Mode 3: Selective Pipeline Runs
```bash
# Skip transcription if already done (expensive API call)
python test2.py             # Evaluate existing transcripts
python test4.py             # Re-consolidate results
python test5.py             # Generate new Excel reports

# Just do eye tracking on new videos
python test6.py

# Clean up previous outputs before re-running
python delete.py            # Interactive cleanup utility
```

### Expected Runtimes (per 100 candidates)
| Stage | Duration | Cost |
|-------|----------|------|
| Transcription (test.py) | 30-45 min | $15-20 (OpenAI) |
| LLM Evaluation (test2.py) | 20-30 min | $8-12 |
| Document Creation (test3-4) | 5-10 min | <$1 |
| Excel Generation (test5.py) | 2-3 min | <$1 |
| Eye Tracking (test6.py) | 45-60 min | $0 (local processing) |
| Aggregation (test7-8.py) | 3-5 min | <$1 |
| **Total** | **~2 hours** | **~$25-35** |

---

## Project Structure

```
Transcript/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env                               # Configuration (create from .env.example)
├── .env.example                       # Env template
├── .gitignore                         # Git ignore rules
│
├── 📜 CORE PIPELINE SCRIPTS
├── test.py          (570 lines)       # Video transcription engine
├── test2.py         (569 lines)       # LLM evaluation module
├── test3.py         (322 lines)       # Individual analysis doc generator
├── test4.py         (537 lines)       # Slot-level consolidation
├── test5.py         (662 lines)       # Excel sheet generation
├── test6.py         (1536 lines)      # Eye tracking & gaze analysis engine
├── test7.py         (433 lines)       # Results aggregation
├── test8.py         (476 lines)       # Final delivery packaging
├── delete.py        (627 lines)       # Cleanup & reset utility
│
├── 📋 EVALUATION PROMPTS (LLM Instructions)
├── intro-prompt.txt                   # Introduction video scoring criteria
├── mock-prompt.txt                    # Mock interview evaluation framework
├── project-scenario.txt               # Project explanation assessment
├── niche-prompt.txt                   # Domain-specific knowledge evaluation
├── CV-prompt.txt                      # Resume alignment & credibility check
├── Tools-Technology-prompt.txt        # Technology stack expertise testing
│
├── 📚 REFERENCE MATERIALS
├── 31-Questions.pdf                   # Sample mock interview questions
├── Niche-Questions.pdf                # Domain-specific question database
│
├── 🤖 COMPUTER VISION MODELS
├── models/
│   ├── blaze.onnx          (535 KB)   # Face detection model
│   └── gaze_estimation.onnx (95 MB)   # 3D gaze vector computation
│
├── 🐳 CONTAINER DEPLOYMENT
├── Dockerfile                         # Python 3.11 container image definition
├── docker-compose.yml                 # Multi-stage orchestration config
│
├── 🔐 GOOGLE INTEGRATION
├── credentials.json                   # OAuth credentials (not in repo)
├── token.json                         # OAuth token cache (auto-managed)
│
└── 📁 CACHES & TEMP FILES
    ├── .openai_file_cache.json        # Uploaded file IDs from OpenAI
    └── tmpclaude-*.cwd                # Temporary working directories
```

---

## Evaluation Criteria

### 1️⃣ Introduction Video (intro-prompt.txt)
Evaluates self-introduction quality on:
- **Mandatory Info**: Name, title, years of experience, domain, company
- **Presentation**: Clarity, fluency, professionalism, tone
- **Content**: Relevance to role, credibility assessment
- **Detection**: Identifies scripted vs. authentic responses
- **Threshold**: PASS if score ≥ 40%

### 2️⃣ Mock Interview (mock-prompt.txt)
Assesses Q&A performance against 31 reference questions:
- **Relevance**: Answer addresses the question asked
- **Depth**: Technical understanding & explanation quality
- **Confidence**: Communication clarity & articulation
- **Professionalism**: Industry terminology & etiquette
- **Tolerance**: Accounts for speech-to-text errors
- **Threshold**: PASS if score ≥ 40%

### 3️⃣ Project Scenarios (project-scenario.txt)
Evaluates project explanation & ownership:
- **Understanding**: Deep product/project knowledge
- **Framing**: Problem → Solution → Impact narrative
- **Authenticity**: Detects resume reading vs. real experience
- **Technical Depth**: Architecture & design decisions
- **Challenges**: Discusses trade-offs & learned lessons
- **Threshold**: PASS if score ≥ 40%

### 4️⃣ Domain-Specific Knowledge (niche-prompt.txt)
Tests expertise in candidate's specialty:
- **Reference**: Compared against Niche-Questions.pdf (Java, AI/ML, Python, Finance, etc.)
- **Terminology**: Correct use of domain-specific language
- **Depth**: Goes beyond surface-level knowledge
- **Gap Analysis**: Identifies knowledge weaknesses
- **Threshold**: PASS if score ≥ 40%

### 5️⃣ Resume Validation (CV-prompt.txt)
Verifies resume claims through Q&A:
- **Consistency**: Experience stories match resume timeline
- **Depth**: Can articulate details about listed projects
- **Credibility**: Not just name-dropping, shows ownership
- **Red Flags**: Detects exaggeration or fabrication
- **Impact**: Quantifiable results from claimed work
- **Threshold**: PASS if score ≥ 40%

### 6️⃣ Technology Skills (Tools-Technology-prompt.txt)
Assesses practical tool & framework proficiency:
- **Hands-On**: Distinguishes theoretical vs. real experience
- **Context**: Understanding when to use specific tools
- **Depth**: Goes beyond "used it for a project"
- **Comparison**: Awareness of alternatives & trade-offs
- **Resume Match**: Claimed skills validated through conversation
- **Threshold**: PASS if score ≥ 40%

---

## Output Formats

### 1. Individual Analysis Documents (Google Docs)
**Location**: `2026/<Slot>/<Candidate>/Deliverables Analysis/`

```
CANDIDATE EVALUATION REPORT
Slot: [Slot Name] | Candidate: [Name] | Date: [ISO Date]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. INTRODUCTION VIDEO ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LLM-generated detailed evaluation with score: X/10]

2. MOCK INTERVIEW PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LLM analysis of Q&A quality with score: X/10]

3. PROJECT SCENARIO EXPLANATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Assessment of technical depth & ownership: X/10]

4. DOMAIN-SPECIFIC KNOWLEDGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Expertise evaluation against niche criteria: X/10]

5. RESUME VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Consistency & credibility analysis: X/10]

6. TECHNOLOGY SKILLS ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Technical proficiency evaluation: X/10]

SUMMARY
───────────────────────────────────────────────────────────
Overall Score: X% | Status: PASS/FAIL
Eye Contact Score: X% | Status: PASS/FAIL
Recommendations: [...]
```

### 2. Slot-Level Consolidated Report (Google Doc)
**Location**: `Candidate Result/<Slot>/All Deliverables Analysis`

Aggregated assessments for all candidates with:
- Comparative scoring across candidates
- Top performers highlighted
- Recommendation rankings
- Accessible to hiring team (rajvi, sahil, soham@techsarasolutions.com)

### 3. Excel Analysis Sheets
**Location**: `Candidate Result/<Slot>/Deliverables Analysis Sheet.xlsx`

**Worksheets**:
- `Summary`: Candidate rankings by overall score
- `Deliverables`: Scores for each 6-dimensional assessment
- `Eye Tracking`: Gaze metrics (eye contact %, duration, consistency)
- `Detailed Scores`: Per-question/per-video breakdown
- `Pass/Fail`: Threshold compliance matrix

**Columns**:
| Rank | Candidate | Overall | Intro | Mock | Project | Niche | CV | Tools | Eye % | Status |
|------|-----------|---------|-------|------|---------|-------|----|----|------|--------|
| 1 | John Doe | 78% | 8 | 8 | 7.5 | 8 | 8 | 7.5 | 85% | ✓ PASS |

### 4. Eye Tracking Outputs (test6.py)
**Per Video Outputs**:
- `__EYE_result.json` - Detailed frame-by-frame metrics
- `__EYE_annotated_h264.mp4` - Annotated video with gaze visualization
- `__EYE_metrics.csv` - Aggregated statistics

**CSV Columns**:
```
frame_number,timestamp,eye_contact_confidence,gaze_direction_x,gaze_direction_y,
gaze_direction_z,head_position_x,head_position_y,head_position_z,
left_eye_openness,right_eye_openness,blink_detected
```

### 5. Transcript Outputs (test.py)
**Format**: Text files with speaker diarization
**Location**: Original video folders in Google Drive
**Naming**: `transcript.txt`, `LLM_OUTPUT__[dimension].txt`

---

## Docker Deployment

### Build & Run Locally
```bash
# Build image
docker build -t techsara-eval:latest .

# Run container
docker-compose up

# Or single command
docker run --rm -v $(pwd):/workspace techsara-eval:latest python test7.py
```

### Deploy to Cloud

#### AWS ECS
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT].dkr.ecr.us-east-1.amazonaws.com
docker tag techsara-eval:latest [ACCOUNT].dkr.ecr.us-east-1.amazonaws.com/techsara-eval:latest
docker push [ACCOUNT].dkr.ecr.us-east-1.amazonaws.com/techsara-eval:latest

# Create ECS task definition with:
# - Volume mounts for .env, credentials.json
# - Environment variables: SLOT_CHOICE, OPENAI_API_KEY
# - Memory: 2GB+, CPU: 1+ vCPU
# - Timeout: 4+ hours
```

#### AWS Lambda (Smaller Batches)
```bash
# Not ideal for full pipeline (4-hour timeout limit)
# Better for: test7.py (results aggregation), test8.py (delivery)
```

#### Google Cloud Run
```bash
# Build & deploy
gcloud run deploy techsara-eval \
  --source . \
  --memory 2Gi \
  --timeout 3600 \
  --set-env-vars OPENAI_API_KEY=sk-proj-...
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. `OpenAI API Rate Limit (429)`
**Symptom**: `RateLimitError: Error code: 429`

**Solution**:
- Exponential backoff is built-in, but wait 5+ minutes before retry
- Check API usage at [platform.openai.com/account/usage](https://platform.openai.com/account/usage)
- Consider upgrading to paid tier (higher rate limits)
- Stagger batch runs: avoid processing multiple slots simultaneously

#### 2. `Google OAuth Token Expired`
**Symptom**: `google.auth.exceptions.RefreshError`

**Solution**:
```bash
# Delete token and re-authenticate
rm token.json
python test.py  # Will prompt for OAuth consent
```

#### 3. `Video File Not Found in Google Drive`
**Symptom**: `FileNotFoundError` or script skips specific candidates

**Solution**:
- Verify folder structure matches expected naming
- Check Google Drive sharing permissions
- If using Shared Drives, set `USE_SHARED_DRIVES=true` in .env
- Ensure credentials have proper scopes (Drive, Docs, Sheets)

#### 4. `Memory Error During Eye Tracking (test6.py)`
**Symptom**: `MemoryError` or Out of Memory

**Solution**:
- Reduce batch size by running specific slots at a time
- Close other applications
- Increase system RAM or use higher-spec hardware
- Split large videos into multiple files

#### 5. `OpenAI Model Not Found`
**Symptom**: `NotFoundError: The model gpt-5 does not exist`

**Solution**:
- Use available models: `gpt-4-turbo`, `gpt-4o`, `gpt-3.5-turbo`
- Update `OPENAI_MODEL` and `OPENAI_MODEL_MINI` in .env
- Check [OpenAI Models API](https://platform.openai.com/docs/models)

#### 6. `Google Docs API Quota Exceeded`
**Symptom**: `HttpError 403: The caller does not have permission`

**Solution**:
- Google Docs API has rate limit (~300 requests/minute)
- Wait 10 minutes and retry
- Batch operations are queued internally in test4.py

#### 7. `Gaze Estimation Model Not Found`
**Symptom**: `FileNotFoundError: gaze_estimation.onnx`

**Solution**:
```bash
# Download ONNX models manually
mkdir -p models/
# Download from: [model repository]
# Place gaze_estimation.onnx (95 MB) in models/ directory
```

---

## AWS Automation & Cloud Deployment Guide

As a **senior full-stack engineer with AWS experience**, here's the recommended cloud setup:

### Architecture
```
                        ┌─────────────────┐
                        │  Google Drive   │
                        │  (Video Input)  │
                        └────────┬────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  AWS EventBridge Rule    │
                    │  (Daily @ 8 AM UTC)      │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  AWS Lambda (Orchestrator)    │
                │  - Checks for new videos      │
                │  - Triggers ECS Batch Job     │
                └────────────────┬──────────────┘
                                 │
            ┌────────────────────▼──────────────────┐
            │       AWS ECS Batch (Compute)         │
            │  - Docker container orchestration     │
            │  - test.py → test8.py pipeline        │
            │  - Parallel processing per slot       │
            │  - Spot instances (cost optimization) │
            │  - CloudWatch logging                 │
            └────────────────────┬──────────────────┘
                                 │
                    ┌────────────▼──────────┐
                    │  AWS S3 (Cache Layer) │
                    │  - Transcripts        │
                    │  - Analysis results   │
                    │  - Eye tracking data  │
                    └───────────────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  Google Drive + Docs      │
                    │  (Final Output Delivery)  │
                    └───────────────────────────┘
```

### Setup Steps

#### 1. Create AWS IAM Role
```bash
aws iam create-role --role-name techsara-eval-role \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --role-name techsara-eval-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

#### 2. Push Docker Image to ECR
```bash
aws ecr create-repository --repository-name techsara-eval --region us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT].dkr.ecr.us-east-1.amazonaws.com
docker build -t techsara-eval:latest .
docker tag techsara-eval:latest [ACCOUNT].dkr.ecr.us-east-1.amazonaws.com/techsara-eval:latest
docker push [ACCOUNT].dkr.ecr.us-east-1.amazonaws.com/techsara-eval:latest
```

#### 3. Create ECS Task Definition
```json
{
  "family": "techsara-eval-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "techsara-eval",
    "image": "[ACCOUNT].dkr.ecr.us-east-1.amazonaws.com/techsara-eval:latest",
    "essential": true,
    "environment": [
      {"name": "OPENAI_API_KEY", "value": "sk-proj-..."},
      {"name": "SLOT_CHOICE", "value": "1"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/techsara-eval",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    },
    "mountPoints": [
      {"sourceVolume": "credentials", "containerPath": "/root/credentials", "readOnly": true}
    ]
  }],
  "volumes": [
    {"name": "credentials", "efsVolumeConfiguration": {"filesystemId": "fs-xxxxx"}}
  ]
}
```

#### 4. Create EventBridge Scheduler
```bash
aws events put-rule --name techsara-daily-schedule \
  --schedule-expression 'cron(0 8 * * ? *)' \
  --description "Daily candidate evaluation trigger"

aws events put-targets --rule techsara-daily-schedule \
  --targets "Id"="techsara-lambda","Arn"="arn:aws:lambda:us-east-1:[ACCOUNT]:function:techsara-orchestrator"
```

#### 5. Lambda Orchestrator Function
```python
# lambda_handler.py
import boto3
import json
from datetime import datetime

ecs_client = boto3.client('ecs')

def lambda_handler(event, context):
    """Orchestrate ECS batch job for candidate evaluation"""

    response = ecs_client.run_task(
        cluster='techsara-eval-cluster',
        taskDefinition='techsara-eval-task',
        launchType='FARGATE',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-xxxxx'],
                'securityGroups': ['sg-xxxxx'],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [{
                'name': 'techsara-eval',
                'environment': [
                    {'name': 'SLOT_CHOICE', 'value': '1'},
                    {'name': 'TIMESTAMP', 'value': datetime.utcnow().isoformat()}
                ]
            }]
        }
    )

    return {
        'statusCode': 200,
        'body': json.dumps(f"ECS Task started: {response['tasks'][0]['taskArn']}")
    }
```

#### 6. CloudWatch Monitoring
```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/techsara-eval

# Create alarms
aws cloudwatch put-metric-alarm \
  --alarm-name techsara-task-failed \
  --alarm-actions arn:aws:sns:us-east-1:[ACCOUNT]:alerts \
  --metric-name TasksFailed \
  --namespace AWS/ECS \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold
```

### Cost Optimization
- **Spot Instances**: Save 70% on compute (ECS Spot)
- **S3 Intelligent-Tiering**: Auto-archive old results
- **Reserved Capacity**: For 24/7 baseline
- **Budget**: ~$200-300/month for typical usage

---

## Contributing

### Development Workflow
1. Create feature branch: `git checkout -b feature/evaluation-improvement`
2. Make changes and test locally
3. Ensure pipeline runs end-to-end without errors
4. Submit PR with description of changes

### Testing
```bash
# Test individual components
python -m pytest tests/test_transcription.py -v
python -m pytest tests/test_evaluation.py -v
python -m pytest tests/test_eye_tracking.py -v

# Full integration test
python test.py --dry-run  # Preview without API calls
```

### Code Standards
- Python 3.10+
- Type hints where applicable
- Comprehensive error handling
- CloudWatch logging integration
- AWS best practices compliance

---

## License

**Proprietary - TechSara Solutions**

Internal use only. Unauthorized distribution prohibited.

---

## Support & Contact

For issues, questions, or feature requests:
- **Engineering Team**: [engineering@techsarasolutions.com]
- **Documentation**: See inline code comments & docstrings
- **Bug Reports**: Create GitHub Issue with:
  - Error message & stack trace
  - Steps to reproduce
  - Environment (OS, Python version, Docker version)

---

## Appendix: API Cost Estimation

For **100 candidates** in a batch evaluation:

| Service | Usage | Cost |
|---------|-------|------|
| **OpenAI Transcription** | 100 videos × 30min avg | $15-20 |
| **OpenAI GPT-4** | 600 LLM calls (6 per candidate) | $8-12 |
| **Google Drive API** | 1000+ requests | $0 (free tier) |
| **Google Docs API** | 200-300 requests | $0 (free tier) |
| **AWS ECS** (4 hours compute) | m5.large Spot × 4hrs | $0.20-0.30 |
| **Total** | | **~$25-35** |

**ROI**: Manual review of 100 candidates = ~2-3 work weeks × $50/hr = $4,000-6,000
This automation: ~$30 + 2 hours supervision = **99% cost reduction**

---

**Last Updated**: February 2026
**Version**: 1.0
**Maintained By**: TechSara Engineering Team
