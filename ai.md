# Examiney.AI: Technical Project Overview

Examiney.AI is an end-to-end, multi-modal AI platform designed to transform the hiring process from a subjective, manual task into an objective, data-driven science. By integrating computer vision, physiological signal processing, and large language models, the system creates a "Digital Candidate Twin"—a comprehensive profile that quantifies technical skill, behavioral traits, and engagement.

## 1. Core Philosophy
Traditional interviews are prone to human bias and inconsistent evaluation criteria. Examiney.AI solves this by ensuring every candidate is measured against the same data-driven benchmarks. The platform analyzes not just what a candidate says, but how they interact, their physiological responses under stress, and their focus levels.

## 2. System Architecture
The project is built on a modern, decoupled stack designed for high-performance AI processing:

*   **Backend:** A FastAPI (Python 3.11) server acts as the central intelligence hub. It orchestrates complex background tasks, manages LLM integrations, and serves a RESTful API.
*   **Frontend:** A Next.js 14 application providing two distinct user experiences: a sleek Recruiter Dashboard for management and a distraction-free, fullscreen Candidate Portal for interviews.
*   **Database:** Supabase (PostgreSQL) handles structured data, session management, and stores the results of the multi-modal analysis in JSONB formats.
*   **Media Storage:** Cloudinary serves as a specialized CDN for audio and video assets, enabling secure, deterministic media handling without consuming local server storage.

## 3. The Multi-Modal Intelligence Stack
The platform's uniqueness lies in its ability to fuse multiple streams of data:

### A. Physiological Intelligence (rPPG)
Using Remote Photoplethysmography (rPPG), the system extracts the candidate's heart rate (BPM) and Heart Rate Variability (HRV) directly through a standard webcam. It utilizes the CHROM algorithm for signal decomposition and a trained spectral MLP (HRNet) to eliminate motion artifacts, providing insight into the candidate's stress levels and emotional regulation.

### B. Vision & Focus Analysis
*   **Gaze Tracking:** Utilizing MediaPipe FaceMesh, the system tracks iris movement. A personalized 15-point affine transform calibration accounts for individual differences and screen sizes.
*   **Cheating Detection:** A 9-signal FFT-based detector analyzes gaze patterns to identify "robotic" reading behaviors or horizontal scan patterns indicative of reading off-screen notes.
*   **Emotion Analysis:** DeepFace classifies facial expressions into 8 distinct categories, mapping the candidate's emotional journey throughout the interview.

### C. Linguistic & Technical Evaluation
*   **Speech-to-Text:** OpenAI Whisper transcribes responses locally, with specialized filters to detect hallucinations or garbage phrases.
*   **Semantic Scoring:** SentenceTransformers compute the similarity between the candidate's transcript and the generated "ideal answer."
*   **LLM Judging:** Google Gemini Flash (with Ollama fallback) performs stage-aware marking, evaluating responses based on technical depth, behavioral STAR methods, or logical reasoning.

### D. Psychological Profiling (OCEAN)
The system aggregates all signals into the "Big Five" personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism). This is depth-weighted, ensuring that short or shallow responses do not skew the final personality profile.

## 4. Key Workflows

### The Recruiter Setup
The workflow begins with the recruiter uploading a Job Description and a candidate's resume. The system uses IBM Docling to parse complex PDF layouts into structured Markdown. Gemini then generates a contextual interview script—typically 18 to 20 questions—specifically tailored to the candidate's past projects and the requirements of the role.

### The Candidate Portal
Candidates log in using one-time credentials. Before the interview begins, they undergo a calibration phase to establish a baseline for gaze and physiological signals. The interview itself is a timed, proctored environment where the system captures high-fidelity audio, video, and iris landmarks for every response.

### The Analysis Pipeline
Once the interview is submitted, a background daemon thread triggers the processing pipeline:
1.  **Transcription:** Converting raw audio to text.
2.  **Scoring:** Scoring dimensions like Communication, Authenticity, and Technical Skill.
3.  **Finalization:** Aggregating all data into the final OCEAN report and calculating a "Job Fit %" based on semantic overlap with the Job Description.
4.  **Reporting:** The "Digital Candidate Twin" is generated, providing the recruiter with radar charts, sentiment timelines, and video playback with overlaid AI insights.

## 5. Security & Integrity
Integrity is maintained through several layers:
*   **Credential Masking:** Candidate passwords are one-time use and hashed using bcrypt.
*   **Fullscreen Enforcement:** The portal prevents candidates from navigating away or opening other tabs during the session.
*   **hallucination Guards:** Speech-to-text outputs are validated to ensure AI-generated "filler" doesn't affect the candidate's score.
```
