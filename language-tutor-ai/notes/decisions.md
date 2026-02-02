# Architectural Decisions

## Decision 001 – LoRA-first + RAG
We fine-tune a small LLM using LoRA to enforce:
- pedagogical behavior
- output structure
- bilingual consistency

All factual / methodological knowledge is retrieved via RAG.

Date: 2026-01-31
