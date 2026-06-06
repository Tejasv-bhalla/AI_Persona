# Evals Report — Tejasv Bhalla AI Persona
**Date:** June 7, 2026 | **System:** RAG Portfolio Chatbot & Voice Booking Agent

---

## 🎙️ VOICE QUALITY
* **First-Response Latency:** Median **1,425.00 ms** across production calls. Measured from the end of user utterance to the first audio byte sent. STT (Deepgram transcription) median latency was **222.00 ms**, TTS (Cartesia audio synthesis) median latency was **186.00 ms**, and the local LLM streaming generation median latency was **981.22 ms**.
* **Transcription Accuracy:** **96.5%** accuracy (Word Error Rate: **3.5%**) across spoken test script phrases. Evaluated by matching final transcribed JSON logs from Deepgram Nova-2 against the spoken script.
* **Task Completion Rate:** **100% (5/5 calls)** successfully negotiated booking slots, extracted verbal names/emails, and triggered the backend `/book` webhook to generate confirmed calendar invites in Cal.com.
* **Pipeline Robustness:** **3/3 barge-in tests** succeeded (Vapi immediately stopped playback on user speech interruption); **3/3 adversarial tests** successfully refused prompt injections and ChatGPT roleplays.

---

## 💬 CHAT GROUNDEDNESS & RETRIEVAL
* **Hallucination Rate:** **0% (0/10 responses flagged)**. Measured using a custom judge model (`gemini-3.1-flash-lite`) evaluating actual chatbot answers against the ground truth on a 10-question Golden Q&A set.
* **Correctness Breakdown:** **6/10 (60.0%)** responses classified as correct, **2/10 (20.0%)** as partial (omitting minor technical name specs), and **2/10 (20.0%)** as polite refusals/redirects for out-of-scope topics.
* **Retrieval Quality (Own Corpus):**
  * **Faithfulness:** **0.97 / 1.00** (Indicates answers are strictly grounded in retrieved documents).
  * **Answer Relevancy:** **0.67 / 1.00** (Measures how directly the responses address the user's queries).
  * **Context Precision:** **0.55 / 1.00** (Measures the ratio of relevant chunks retrieved in top results).
  * **Context Recall:** **0.57 / 1.00** (Measures if all necessary facts to answer the query were successfully retrieved).
* **Chat Latencies:** Median streaming latency was **981.22 ms** | P95 latency was **1,786.40 ms** (local execution).

---

## 🚧 DEVELOPMENT HURDLES: THE FREE-TIER STACK LIMITATION
* **Render RAM Ceiling (ONNX OOM Crashes):**
  Our main container was limited to Render’s free tier **512MB RAM ceiling**. Integrating the FlashRank cross-encoder (ONNX format) consumed 150-220MB, consistently crashing the server. Resolved by replacing FlashRank with a lightweight, direct NumPy cosine similarity calculation.
* **Groq Daily Token Quotas (TPD 429s):**
  Exhausted the 100k daily token limit on Groq’s Llama 3.3 70B model during testing. Resolved by implementing a custom stream-iteration exception handler that falls back to Llama 3.1 8B mid-conversation without dropping the call.
* **Gemini Call Rate Limits (15 RPM Ceiling):**
  Gemini 3.1 Flash Lite blocked evaluations with 429s when running Ragas. Resolved by implementing a 4.5-second spacing rate-limiter and running Ragas evaluation tasks in a single-threaded queue.

---

## 🛠️ FAILURE MODES & FIXES
1. **Self-Ingestion Bug:**
   * *Problem:* Chatbot answered questions about Tejasv's resume with its own source code definitions.
   * *Root Cause:* The ingestion pipeline lacked a blocklist filter, causing it to discover and index the `AI_Persona` codebase.
   * *Fix:* Added `repo_blocklist` to `Settings` and modified `discover_repositories` to skip indexing code files for blocklisted repos.
2. **Shramik.AI Skipping Bug:**
   * *Problem:* Ingestion pipeline silently skipped indexing the external `Shramik.AI` repository.
   * *Root Cause:* The repository was missing a `CONTRIBUTION-SCOPE.md` file, which caused the pipeline to skip it without alerts.
   * *Fix:* Added an explicit runtime check that halts ingestion with clear instructions if a required scope file is missing.
3. **Intent Misclassification Bug:**
   * *Problem:* Queries like *"Can you tell me about your education?"* were routed as small talk instead of performing database retrieval.
   * *Root Cause:* The guard model's prompt was biased toward classifying conversational greetings as small talk regardless of content.
   * *Fix:* Updated the `GUARD_PROMPT` to prioritize professional keywords (e.g. education, projects) as `rag` regardless of conversational tone.

---

## ⚖️ CONSCIOUS TRADE-OFF
* **FlashRank Reranking Removal (Accuracy vs. System Stability/Cost):**
  We eliminated the FlashRank cross-encoder reranker from the retrieval pipeline, relying solely on direct NumPy-based cosine similarity. FlashRank’s 150-220MB ONNX model memory footprint, combined with the FastEmbed embedding pipeline, exceeded Render’s free tier **512MB RAM ceiling**, causing Out-Of-Memory (OOM) container crashes. Removing FlashRank traded **3–7% retrieval precision** but kept the backend server 100% stable on lightweight, free hosting.

---

## 🚀 2-WEEK ROADMAP
* **Week 1: Direct Twilio WebSockets Audio Pipeline**  
  Transition from Vapi’s managed telephony wrapper to a direct **Twilio Media Streams WebSocket integration**. While Vapi got us running quickly, a direct Twilio pipeline provides low-level control over raw audio bytes, custom barge-in sensitivity, and custom SIP routing, eliminating the third-party platform dependency.
* **Week 2: MD5 Incremental Ingestion & Automated CI/CD Evals**  
  Implement chunk-level MD5 hashing to avoid full database rebuilds (updating only modified files in under 60 seconds), and configure automated Ragas runs via GitHub Actions to block code deployments if the custom judge hallucination rate rises above 0%.
