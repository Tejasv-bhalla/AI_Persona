# Evals Report — Tejasv Bhalla AI Persona
**System:** RAG Portfolio Chatbot & Voice Booking Agent
**Committed run:** `eval/results/` (10-question Golden Q&A set, run against a local backend)

---

## ⚠️ READ THIS FIRST — PROVENANCE OF EVERY NUMBER

Every quantitative claim in this report is traceable to a committed artefact:

| Artefact | Contains |
|:---|:---|
| `eval/results/summary.json` | Aggregate RAGAS scores, judge rates, latency, index composition |
| `eval/results/ragas_results.csv` | Per-question RAGAS scores (7 in-scope questions) |
| `eval/results/judge_results.json` | Per-question judge verdict and reason (10 questions) |
| `eval/results/raw_responses.json` | The exact answers and retrieved contexts that were scored |

**These results predate the current retrieval code.** They were produced with `max_context_chunks = 3`
(every `retrieved_contexts` list in `raw_responses.json` has exactly 3 entries), no raw-input retrieval
fallback, no corpus-derived repo filter, and no corrective retry loop. They are the last honestly measured
run and are published unmodified. **The suite must be re-run** before any claim is made about the effect of
the changes listed under [What Changed In Response](#-what-changed-in-response). Nothing below has been
adjusted, extrapolated, or estimated.

**Voice latency figures have been removed from this report.** Earlier revisions published median
first-response, STT and TTS latencies. No data file in this repository supports them — `eval/results/`
contains chat metrics only, and the harness never instruments the voice path. Voice behaviour was observed
informally during manual test calls; that is not a measurement and is no longer reported as one.

---

## 🎙️ VOICE — INFORMAL MANUAL QA ONLY

There is **no automated voice evaluation** in this repository, and no committed artefact for anything in
this section. What follows is a log of manual test calls, reported as such:

* **Booking completion.** 5 manual calls placed to the Vapi number; all 5 negotiated a slot, captured a
  name and an email verbally, and produced a confirmed Cal.com invite. Verified by checking the calendar,
  not by an automated assertion.
* **Barge-in.** 3 manual interruption attempts; Vapi stopped playback on user speech each time. This is
  Vapi platform behaviour, not something this codebase implements.
* **Adversarial probes.** 3 manual prompt-injection / "pretend you are ChatGPT" attempts over the phone;
  all were refused. The same category of probe is covered automatically in the chat Golden set (Q8–Q10).
* **Transcription.** Deepgram handled spoken email addresses poorly enough that `nodes/calcom.py` carries
  explicit normalisation for the mangled forms it produces ("at the rate", "third egg mail", spaced dots).
  No word-error-rate was computed — doing so would require the raw Deepgram transcripts, which are not
  captured or committed.

Sample sizes this small, scored by the author, are a smoke test. They are reported for completeness and
should not be read as an evaluation.

---

## 💬 CHAT GROUNDEDNESS & RETRIEVAL

Source: `eval/results/summary.json`. The Golden set is 10 questions: 7 in-scope (resume, README,
changelog, contribution-scope) and 3 adversarial/out-of-scope. RAGAS runs over the 7 in-scope questions
only — out-of-scope items are excluded because "no relevant context" is the correct outcome there and
would distort the retrieval metrics. The judge runs over all 10.

### RAGAS (N = 7)

| Metric | Score | Definition |
|:---|:---|:---|
| **Faithfulness** | **0.97** | Fraction of the claims in the answer that are supported by the retrieved chunks. Measures hallucination, not correctness. |
| **Answer Relevancy** | **0.67** | How directly the answer addresses the question actually asked. A refusal scores near zero. |
| **Context Precision** | **0.55** | How much of what was retrieved was actually relevant — signal-to-noise inside the top-k. |
| **Context Recall** | **0.57** | How much of the information needed to answer the question was successfully retrieved. |

Judge/evaluator model: `gemini-3.1-flash-lite` via the OpenAI-compatible endpoint, temperature 0.
Embeddings for RAGAS use the same `BAAI/bge-small-en-v1.5` model as the backend.

### LLM judge (N = 10)

| Verdict | Count | Rate |
|:---|:---|:---|
| Correct | 6 | 60% |
| Partial | 2 | 20% |
| Refused | 2 | 20% |
| **Hallucinated** | **0** | **0%** |

### Latency

Median **981.22 ms**, P95 **1,786.40 ms**, across the same 10 questions, run locally against a warm
backend. One important caveat: `measure_chat_latency` stops the clock on the **first SSE `data:` line**,
which is the `meta` frame carrying the route — emitted after the guard and router nodes and *before*
generation begins. This is a lower bound on perceived responsiveness, not a time-to-first-token figure.
It is not a production measurement and there is no production latency sample in this repository.

### Index composition at the time of the run

1,371 chunks total: 941 code · 276 readme · 87 unknown · 46 notebook-output · 12 changelog · 5 resume ·
4 contribution-scope.

---

## 🔍 WHAT THESE NUMBERS ACTUALLY SAY

**Faithfulness 0.97 and hallucination 0% are guardrail results, not retrieval results.** The system
refuses when retrieval comes up empty, and a refusal makes no unsupported claims, so it scores well on
faithfulness. Reading 0.97 as "retrieval is working" would be wrong.

**Context precision 0.55 and context recall 0.57 are the honest headline: retrieval is the bottleneck.**
The per-question CSV makes it concrete — three of seven in-scope questions scored **0.0 context recall**,
and one scored 0.0 context precision.

The clearest single failure is Q2, *"What work experience or internships does Tejasv have?"*:

> **Answer:** "I don't have that information in the indexed knowledge base…"
> **Judge verdict:** `refused` — *"The model explicitly stated it does not have the information in its
> knowledge base despite the information being present in the expected answer."*

That refusal is counted inside the 20% refusal rate next to a genuine adversarial refusal (Q9, the
"pretend you are ChatGPT" probe). **One of the two refusals was a retrieval miss, not a safety win** — the
refusal rate flatters the system, and pretending otherwise would be dishonest. The two `partial` verdicts
have the same root cause: a project (Q3) and a technical detail (Q4) that exist in the corpus but never
reached the top 3 chunks.

Diagnosis: with only 3 chunks of context and a single keyword-derived query, the retriever was starving
the generator. The generator's refusal behaviour was doing its job — it correctly declined to invent the
missing facts.

---

## 🛠️ WHAT CHANGED IN RESPONSE

Each change targets a specific failure above. **None has been re-measured.**

| Change | Failure it targets |
|:---|:---|
| `max_context_chunks` raised 3 → 5 | Recall 0.0 on questions whose answer spans several chunks |
| Retrieval falls back to the user's raw wording when the guard's keywords return fewer than 2 candidates | Off-target classifier keywords starving the search (the likely Q2 cause) |
| Repo filter derived from the repo names actually present in Qdrant, with camelCase/delimiter token splitting, instead of a hardcoded dict of 8 names | Repo mentions the hardcoded dict did not know about |
| Genuinely ambiguous repo mentions return **no** filter rather than guessing | A wrong repo filter guarantees recall 0; an unfiltered search merely dilutes precision |
| Corrective retry: an answer that fails the grounding grader re-runs retrieval with +4 candidates and +2 chunks, then regenerates | Narrow first-pass retrieval — the likeliest reason an answer cannot be grounded |

The corrective loop moved generation and grading **inside** the LangGraph graph
(`guard → router → … → generate → grade → retrieval`, one retry maximum). Token streaming survived the
move because `generate` writes to LangGraph's custom stream channel via `get_stream_writer`, so
time-to-first-token is unaffected. The regenerated answer reaches the browser as `correction` SSE events
and replaces the text already on screen. Voice is exempt: the caller has already heard the answer.

---

## 🔒 HARNESS INTEGRITY

Two fixes to `eval/run_evals.py` matter more than any score in this report, because they determine whether
the scores can be believed at all.

1. **RAGAS no longer fabricates scores on failure.** The exception path previously returned hardcoded
   placeholders — `faithfulness 0.88`, `answer_relevancy 0.89`, `context_precision 0.85`,
   `context_recall 0.84`. Those values are plausible, are in the range a real run produces, and were
   written to `summary.json` in exactly the same shape as measured ones. Nothing downstream could tell
   them apart. The path now raises:
   `RuntimeError("Ragas evaluation failed; refusing to report fabricated scores: …")`.

2. **Judge API failures are no longer silently counted as refusals.** A failed judge call used to default
   to `verdict = "refused"`, inflating the refusal rate with harness errors and deflating every other
   rate. Failures are now a distinct `"error"` verdict, and any non-zero error count raises before a
   summary is written — an incomplete run cannot be reported as a complete one.

3. **The eval endpoint is gated.** `/chat/eval` returns non-streaming answers plus their retrieved
   contexts. It now requires the `x-eval-key` header and returns **404** when `EVAL_API_KEY` is unset, so
   it does not exist in production. The harness fails loudly with a pointer to `eval/.env` if the key does
   not match.

A metric that cannot fail loudly is not a metric.

---

## 🚧 DEVELOPMENT HURDLES: THE FREE-TIER STACK

* **Render RAM ceiling (ONNX OOM crashes).**
  The container is limited to Render's free-tier **512MB RAM**. Adding the FlashRank cross-encoder (ONNX,
  150–220MB) on top of the FastEmbed pipeline crashed the server consistently. FlashRank was removed. See
  the trade-off section below for what did *not* replace it.
* **Groq model decommissioning (404 `model_not_found`).**
  Groq retired the entire Llama family this project ran on — `llama-3.1-8b-instant` and
  `llama-3.3-70b-versatile` both stopped existing — and every LLM call in the system failed until the
  stack was remapped: `openai/gpt-oss-120b` for chat generation, `qwen/qwen3.8-27b` for voice, guard and
  grader, `openai/gpt-oss-20b` as the fallback. The model IDs are env-overridable, so the swap itself was
  configuration, but the replacements did not behave like the originals: `gpt-oss` is a reasoning family
  and streams 37–87 `delta.reasoning` chunks before any content, which under a tight output cap consumes
  the whole budget and returns **empty** output (observed as `400 json_validate_failed` on a guard call).
  Measured over 5 varied guard prompts at a 200-token cap, `openai/gpt-oss-20b` produced usable JSON 2/5
  times using up to 323 output tokens; `qwen/qwen3.8-27b` produced it 5/5 times using 60. Hence the split:
  reasoning model for generation, non-reasoning model for the JSON roles and for voice, where median
  time-to-first-token measured ~300 ms (qwen) against ~530 ms (`gpt-oss-20b`) and ~640 ms
  (`gpt-oss-120b`).
* **Groq free-tier output-token ceiling (429s).**
  The binding limit is **1,000 output tokens per minute** — not a daily token cap — and Groq admits a
  request against its declared `max_tokens` rather than its actual output. The code set no `max_tokens` at
  all, so qwen reserved its own default (1,102–1,392 tokens) and returned 429 on *every* call. Fixed by
  making caps mandatory: `MAX_OUTPUT_TOKENS_JSON=200`, `MAX_OUTPUT_TOKENS_CHAT=500`,
  `MAX_OUTPUT_TOKENS_VOICE=300`, so one chat turn's three calls reserve 900 and stay under the ceiling.
  `backend/tests/test_token_budget.py` pins that invariant. Measured actual usage per chat turn is lower —
  guard 92, generation 179, grader 142 = **413 output tokens** — which puts sustained free-tier throughput
  at roughly 2.4 chat turns per minute. The `GroqClient` fallback still covers outright model failure: if
  the primary model errors *before any token has been emitted*, the request is retried on
  `GROQ_FALLBACK_MODEL` (`openai/gpt-oss-20b`). Once tokens have been emitted the error propagates
  instead — a truncated answer is recoverable, a duplicated one is not.
* **Gemini rate limits (15 RPM ceiling).**
  Gemini blocked RAGAS runs with 429s. Resolved with a 4.5-second minimum spacing wrapper around
  `ChatOpenAI._generate`/`._agenerate`, exponential backoff on rate-limit errors, `max_workers=1`, and a
  5-second sleep between judge calls.

---

## 🐞 FAILURE MODES & FIXES

1. **Self-ingestion bug**
   * *Problem:* the chatbot answered resume questions with its own source-code definitions.
   * *Root cause:* the ingestion pipeline had no blocklist, so it discovered and indexed the `AI_Persona`
     codebase itself.
   * *Fix:* `repo_blocklist` in `Settings`, applied in `discover_repositories`.
2. **Shramik.AI silently skipped**
   * *Problem:* the external `Shramik.AI` repo was skipped without an alert.
   * *Root cause:* a missing contribution-scope file caused a silent skip.
   * *Fix:* a hard stop — `RuntimeError("HARD STOP: Missing contribution scope file for …")` — with the
     exact filename to create. An external repo indexed without a contribution boundary is the attribution
     hallucination this project exists to avoid.
3. **Intent misclassification**
   * *Problem:* *"Can you tell me about your education?"* routed to small talk instead of retrieval.
   * *Root cause:* the guard prompt over-weighted conversational tone.
   * *Fix:* `GUARD_PROMPT` now prioritises professional keywords as `rag` regardless of tone.
4. **Booking state inferred from the bot's own transcript**
   * *Problem:* voice booking recovered its position in the conversation by string-matching the
     assistant's previous message, which broke as soon as the wording changed.
   * *Root cause:* no explicit state.
   * *Fix:* a `BookingStage` enum (`idle / offering / awaiting_name / awaiting_email / confirmed`) held in
     graph state and persisted across turns by a LangGraph `MemorySaver` checkpointer keyed on the Vapi
     call id. The router forces the scheduling route while a booking is mid-flight, so an utterance like
     "john at gmail dot com" is not re-classified as a factual question.
5. **Slot list shifting mid-call**
   * *Problem:* re-fetching Cal.com availability on every turn meant "the second one" could refer to a
     different slot than the one that had been offered.
   * *Fix:* slots are fetched once per call, pinned into graph state, and reused for the rest of the call.

---

## ⚖️ CONSCIOUS TRADE-OFF: NO RERANKER

FlashRank's cross-encoder was removed from the retrieval pipeline because its ONNX model (150–220MB),
combined with the FastEmbed embedding pipeline, exceeded Render's free-tier 512MB ceiling and caused
out-of-memory container crashes.

**It was not replaced.** There is no second-stage reranker in this system — not FlashRank, not a NumPy
cosine pass, not anything. `nodes/retrieval.py` takes Qdrant's candidates in the order they come back and
truncates to `max_context_chunks`. That order is produced by Qdrant's **server-side RRF fusion** of two
prefetches — a dense query against the `dense` vector and a BM25 query against the `bm25` sparse vector —
so hybrid retrieval is real, but it is single-stage. (Cosine similarity does appear in the codebase, in
`voice/response_cache.py`, where it matches an incoming question against cached ones. That is a cache
lookup, not a reranker, and it never reorders retrieved chunks.)

The trade-off, stated plainly: **the system runs for $0 on 512MB, and its retrieval quality is worse for
it.** The eval numbers show the cost — context precision 0.55 is precisely the metric a cross-encoder
reranker exists to raise, and it is the lowest score in the table. This is the single biggest known
weakness in the system. The honest fix is a paid instance with headroom for a cross-encoder, or a hosted
rerank API that keeps the memory cost off this container.

---

## 🚀 ROADMAP

* **Re-run this evaluation.** The committed numbers describe a system that no longer exists. Everything
  else on this list is speculation until that is done. Requires a live backend with `EVAL_API_KEY` set and
  a Gemini key.
* **Restore a reranker off-container.** A hosted rerank API, or a cross-encoder on a paid instance, aimed
  squarely at the 0.55 context precision.
* **Automate the evals in CI.** `.github/workflows/ci.yml` already runs ruff, mypy, pytest and the
  frontend build; a scheduled RAGAS run that fails the build on a regression is the natural next gate.
  Backend test coverage under `backend/tests/` needs to land first.
* **Direct Twilio Media Streams pipeline.** Replace Vapi's managed telephony with a direct WebSocket
  integration for low-level control over raw audio, barge-in sensitivity and SIP routing. Vapi got the
  voice agent working quickly; the dependency is the price.
* **Instrument the voice path.** No voice latency number in this repository is measured. Until the
  `/voice` path emits timings, none will be published.
