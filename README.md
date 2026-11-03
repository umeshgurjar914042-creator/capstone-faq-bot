# FAQ Chatbot for [business name / type]

A RAG chatbot that answers customers' common questions from the business's own FAQ, and says "I don't know" when the answer is not there.

## Live demo
[streamlit link]

## Problem
[1-2 lines: what questions were asked repeatedly]

## How it works
1. FAQ lines are embedded and stored in ChromaDB
2. A question finds the closest FAQ line
3. If the distance is above a threshold, the bot says "I don't know" without calling the LLM
4. Otherwise Gemini answers using only the retrieved lines
5. If Gemini is busy, the bot shows the closest FAQ line instead

## Evaluation (small test set)
- Test questions: [N], including paraphrased and out-of-scope ones
- Retrieval hit-rate@1: [X/N], hit-rate@3: [X/N], MRR: [0.XX]
- Threshold [value] chosen from measured distances
- Failed cases and what I learned: [...]

## Tests
pytest: [8] passed (decision logic and fallback)

## Feedback from real users
[what the owner or 2-3 people said, if you tried it]

## Tech
Python, Gemini API, ChromaDB, Streamlit, pytest
