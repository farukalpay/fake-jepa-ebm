# fake-jepa-ebm
Fake JEPA+EBM: joint GloVe and GPT embeddings, cosine-based energy scoring, exposed as a tool for GPT-5.1 responses API.

## Example Output:
```
Context   ['A child is playing with a red ball in the park.']: A child is playing with a red ball in the park.
Candidate ['The kid happily throws the bright red ball across the playground.']: The kid happily throws the bright red ball across the playground.

[DEBUG] Model requested tool `evaluate_joint_embedding_energy` with args:
{
  "context": "A child is playing with a red ball in the park.",
  "candidate": "The kid happily throws the bright red ball across the playground.",
  "verbose": true
}
[INFO] Loading GloVe embeddings from: /Users/farukalpay/Desktop/Glove/glove.6B.300d.txt
[INFO] Loaded 400000 GloVe vectors of dim 300.

[DEBUG] Raw tool result:
{
  "context": "A child is playing with a red ball in the park.",
  "candidate": "The kid happily throws the bright red ball across the playground.",
  "energy": 0.22207340265195397,
  "joint_cosine_similarity": 0.777926597348046,
  "glove_cosine_similarity": 0.8459956381284257,
  "gpt_cosine_similarity": 0.7098574745479543,
  "glove_tokens_context": [
    "a",
    "child",
    "is",
    "playing",
    "with",
    "a",
    "red",
    "ball",
    "in",
    "the",
    "park"
  ],
  "glove_tokens_candidate": [
    "the",
    "kid",
    "happily",
    "throws",
    "the",
    "bright",
    "red",
    "ball",
    "across",
    "the",
    "playground"
  ],
  "glove_dim": 300,
  "gpt_embedding_dim": 1536,
  "interpretation": "Lower energy means the candidate is more compatible with the context under this joint-embedding toy model. This is a simulation of a JEPA + energy-based objective; no training is performed.",
  "norms": {
    "glove_context_norm": 3.9445250034332275,
    "glove_candidate_norm": 3.265620708465576,
    "gpt_context_norm": 0.5040023326873779,
    "gpt_candidate_norm": 0.5121766328811646
  }
}

================ JEPA + EBM MODEL EXPLANATION ================

Energy score: **0.22** (on a scale where **lower = more compatible**)

Explanation:
- The context is: “A child is playing with a red ball in the park.”
- The candidate is: “The kid happily throws the bright red ball across the playground.”

The joint-embedding model computed:
- **Joint cosine similarity:** ~**0.78**  
- Corresponding **energy:** ~**0.22** (energy ≈ 1 − similarity in this toy setup).

What this means:
- The two sentences are **highly semantically compatible**:
  - “child” ↔ “kid”
  - “red ball” ↔ “bright red ball”
  - “in the park” ↔ “across the playground” (closely related setting)
  - Both describe a **playful action with the same object** (the ball).
- The low energy indicates that the candidate is a **natural, coherent continuation or re-description** of the context, with only minor differences in detail and phrasing.

==============================================================
```
