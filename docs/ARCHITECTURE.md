# Architecture

```mermaid
flowchart TD
  A["User input"] --> B["Merged guard: safety + intent + keywords"]
  B --> C["Deterministic router"]
  C -->|scheduling| D["Cal.com tool response"]
  C -->|small talk| E["Small talk response"]
  C -->|RAG| F["FastEmbed query vector"]
  F --> G["Qdrant filtered vector search"]
  G --> H["Hybrid dense + BM25 RRF"]
  H --> L["NumPy cosine rerank"]
  L --> I["Groq streaming generator"]
  I --> J["User sees tokens"]
  I --> K["Async hallucination grader"]
```

## Runtime rule

Raw user input stops at the guard node. The generator sees only sanitized keywords and retrieved context.

## Memory rule

FastEmbed is lazy-loaded. Ingestion is offline. The web container only performs query-time retrieval and generation.

## Grounding rule

If the answer is absent from retrieved context, the persona must say the indexed knowledge base does not contain it.
