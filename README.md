# fake-jepa-ebm
Fake JEPA+EBM: joint GloVe and GPT embeddings, cosine-based energy scoring, exposed as a tool for GPT-5.1 responses API.

## Example Output (main.py):
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

## Example Output (pairwise.py):
```
#JEPA-style PAIRWISE energy model demo.
This demo calculates energy based on all token-token interactions.

Context   ['A child is playing with a red ball in the park.']: 
Candidate ['The kid happily throws the bright red ball across the playground.']: 

[DEBUG] Model requested tool `evaluate_joint_embedding_energy` with args:
{
  "context": "#",
  "candidate": "The kid happily throws the bright red ball across the playground.",
  "verbose": true
}
[INFO] Loading GloVe embeddings from: /Users/farukalpay/Desktop/Glove/glove.6B.300d.txt
[INFO] Loaded 400000 GloVe vectors of dim 300.

[DEBUG] Tool Output (Pairwise Contributors):
{
  "energy": 1.0,
  "mean_pairwise_similarity": 0.0,
  "gpt_global_similarity": 0.1558,
  "top_contributors": [],
  "interpretation": "Energy is calculated from pairwise token interactions. Top contributors list the specific word pairs (h_i, k_j) that had the highest cosine similarity, driving the energy down.",
  "debug_info": {
    "context_tokens_count": 1,
    "candidate_tokens_count": 11,
    "matrix_shape": [
      1,
      11
    ]
  }
}

================ PAIRWISE ENERGY EXPLANATION ================

Energy score: 1.0 (lower is better, so this indicates a weak match between context and candidate).

Explanation:
- The context is just the single character: `#`
- The candidate is: “The kid happily throws the bright red ball across the playground.”
- Because the context has essentially no semantic content, the model can’t find meaningful semantic alignments between words in the context and words in the candidate.

Top contributors (matching word pairs):
- `top_contributors` is empty: `[]`

That means there are no specific word pairs between the context and the candidate that strongly contribute to lowering the energy. In other words, the model does not detect meaningful semantic overlap between `#` and the candidate sentence.

=============================================================

(base) farukalpay@Mac november % python /Users/farukalpay/Desktop/Python/november/main2.py
JEPA-style joint embedding + energy-based model demo.
Press Enter for defaults if you don't feel like typing.

Context   ['A child is playing with a red ball in the park.']: 
Candidate ['The kid happily throws the bright red ball across the playground.']: 

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
  "energy": -0.4228740632534027,
  "energy_definition": "E(x, y) = - sum_{i,j} w_ij * phi(h_i, k_j) over GloVe token embeddings, where phi is cosine similarity and w_ij are softmax-normalized attention weights over all token pairs. Lower energy means a stronger aggregate token\u2013token match between context and candidate.",
  "joint_cosine_similarity": 0.7778803573775505,
  "glove_cosine_similarity": 0.8459956381284257,
  "gpt_cosine_similarity": 0.7097648919017681,
  "legacy_joint_energy": 0.22211964262244954,
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
  "token_pair_contributions_topk": [
    {
      "context_token": "the",
      "candidate_token": "the",
      "phi_cosine": 1.0000001192092896,
      "weight": 0.029797058552503586,
      "contribution": 0.029797062277793884,
      "indices": {
        "i": 9,
        "j": 9
      }
    },
    {
      "context_token": "the",
      "candidate_token": "the",
      "phi_cosine": 1.0000001192092896,
      "weight": 0.029797058552503586,
      "contribution": 0.029797062277793884,
      "indices": {
        "i": 9,
        "j": 4
      }
    },
    {
      "context_token": "the",
      "candidate_token": "the",
      "phi_cosine": 1.0000001192092896,
      "weight": 0.029797058552503586,
      "contribution": 0.029797062277793884,
      "indices": {
        "i": 9,
        "j": 0
      }
    },
    {
      "context_token": "red",
      "candidate_token": "red",
      "phi_cosine": 1.0000001192092896,
      "weight": 0.029797058552503586,
      "contribution": 0.029797062277793884,
      "indices": {
        "i": 6,
        "j": 6
      }
    },
    {
      "context_token": "ball",
      "candidate_token": "ball",
      "phi_cosine": 0.9999999403953552,
      "weight": 0.02979704737663269,
      "contribution": 0.02979704551398754,
      "indices": {
        "i": 7,
        "j": 7
      }
    },
    {
      "context_token": "in",
      "candidate_token": "the",
      "phi_cosine": 0.636134684085846,
      "weight": 0.014392090030014515,
      "contribution": 0.00915530789643526,
      "indices": {
        "i": 8,
        "j": 9
      }
    },
    {
      "context_token": "in",
      "candidate_token": "the",
      "phi_cosine": 0.636134684085846,
      "weight": 0.014392090030014515,
      "contribution": 0.00915530789643526,
      "indices": {
        "i": 8,
        "j": 4
      }
    },
    {
      "context_token": "in",
      "candidate_token": "the",
      "phi_cosine": 0.636134684085846,
      "weight": 0.014392090030014515,
      "contribution": 0.00915530789643526,
      "indices": {
        "i": 8,
        "j": 0
      }
    },
    {
      "context_token": "is",
      "candidate_token": "the",
      "phi_cosine": 0.5431621670722961,
      "weight": 0.01195002906024456,
      "contribution": 0.00649080378934741,
      "indices": {
        "i": 2,
        "j": 9
      }
    },
    {
      "context_token": "is",
      "candidate_token": "the",
      "phi_cosine": 0.5431621670722961,
      "weight": 0.01195002906024456,
      "contribution": 0.00649080378934741,
      "indices": {
        "i": 2,
        "j": 4
      }
    },
    {
      "context_token": "is",
      "candidate_token": "the",
      "phi_cosine": 0.5431621670722961,
      "weight": 0.01195002906024456,
      "contribution": 0.00649080378934741,
      "indices": {
        "i": 2,
        "j": 0
      }
    },
    {
      "context_token": "a",
      "candidate_token": "the",
      "phi_cosine": 0.5241737961769104,
      "weight": 0.011504716239869595,
      "contribution": 0.006030470598489046,
      "indices": {
        "i": 5,
        "j": 0
      }
    },
    {
      "context_token": "a",
      "candidate_token": "the",
      "phi_cosine": 0.5241737961769104,
      "weight": 0.011504716239869595,
      "contribution": 0.006030470598489046,
      "indices": {
        "i": 5,
        "j": 4
      }
    },
    {
      "context_token": "a",
      "candidate_token": "the",
      "phi_cosine": 0.5241737961769104,
      "weight": 0.011504716239869595,
      "contribution": 0.006030470598489046,
      "indices": {
        "i": 5,
        "j": 9
      }
    },
    {
      "context_token": "a",
      "candidate_token": "the",
      "phi_cosine": 0.5241737961769104,
      "weight": 0.011504716239869595,
      "contribution": 0.006030470598489046,
      "indices": {
        "i": 0,
        "j": 0
      }
    },
    {
      "context_token": "a",
      "candidate_token": "the",
      "phi_cosine": 0.5241737961769104,
      "weight": 0.011504716239869595,
      "contribution": 0.006030470598489046,
      "indices": {
        "i": 0,
        "j": 9
      }
    },
    {
      "context_token": "a",
      "candidate_token": "the",
      "phi_cosine": 0.5241737961769104,
      "weight": 0.011504716239869595,
      "contribution": 0.006030470598489046,
      "indices": {
        "i": 0,
        "j": 4
      }
    },
    {
      "context_token": "red",
      "candidate_token": "bright",
      "phi_cosine": 0.5119451284408569,
      "weight": 0.011226754635572433,
      "contribution": 0.005747482180595398,
      "indices": {
        "i": 6,
        "j": 5
      }
    },
    {
      "context_token": "with",
      "candidate_token": "the",
      "phi_cosine": 0.4945622980594635,
      "weight": 0.010843154974281788,
      "contribution": 0.005362615454941988,
      "indices": {
        "i": 4,
        "j": 9
      }
    },
    {
      "context_token": "with",
      "candidate_token": "the",
      "phi_cosine": 0.4945622980594635,
      "weight": 0.010843154974281788,
      "contribution": 0.005362615454941988,
      "indices": {
        "i": 4,
        "j": 4
      }
    }
  ],
  "interpretation": "The main energy is now a sum over all pairwise interactions between GloVe token embeddings from the context and the candidate. Each token\u2013token pair contributes approximately contrib_ij = w_ij * phi(h_i, k_j), where phi is cosine similarity and w_ij is an attention weight derived from a softmax over all pairs. The largest positive contributions indicate the token pairs that most strongly support the match between context and candidate \u2014 these are the 'this is why I think they match' pairs.",
  "norms": {
    "glove_context_norm": 3.9445250034332275,
    "glove_candidate_norm": 3.265620708465576,
    "gpt_context_norm": 0.5040026307106018,
    "gpt_candidate_norm": 0.5121558308601379
  },
  "token_matrix_shapes": {
    "context_tokens": 11,
    "candidate_tokens": 11
  }
}

================ JEPA + EBM MODEL EXPLANATION ================

- The overall energy is **E = -0.423**.  
  - By this model’s convention, **lower (more negative) energy means a stronger match** between context and candidate.  
  - The magnitude (about 0.42 in absolute value) is moderately large given that each individual token–token contribution is on the order of 0.03 or less; aggregating many positive contributions pushes the sum up, and then the minus sign makes the final energy negative.  
  - So the model is saying: these two sentences are **highly compatible**.

- The top token–token contributions tell us *why* the model thinks they match. Each contribution is `w_ij * cosine_similarity`, so higher positive values mean “these two tokens fit together well in context.”

  **Strongest exact-match pairs:**
  - `("the", "the")` multiple times, each with contribution ≈ **0.0298**  
    - Context token: “the” (index 9)  
    - Candidate tokens: “the” at indices 0, 4, 9  
    - These are high because identical words have cosine ≈ 1.0 and get decent attention weight. They show shared function-word structure, giving a broad syntactic compatibility cue, even though “the” is semantically light.
  - `("red", "red")` with contribution ≈ **0.0298**  
    - This is a key **content-word** alignment: both sentences describe a **red** object.
  - `("ball", "ball")` with contribution ≈ **0.0298**  
    - Another core content-word match: both explicitly mention a **ball**.

  These `("red","red")` and `("ball","ball")` pairs are the clearest “this is why I think they match” signals: same object (ball), same color (red).

  **Semantically related content pairs:**
  - `("red", "bright")` with contribution ≈ **0.0057**  
    - “bright” modifies “red” in the candidate. The positive cosine and nontrivial weight indicate the model sees “bright” as semantically reinforcing the color description from the context. This says: *the candidate is elaborating the same visual property (red) rather than contradicting it*.

  **Structural / filler support pairs:**
  - Multiple `("in", "the")` pairs, each contribution ≈ **0.0092**  
    - Context: “in the park”; Candidate: “the playground”. “in” and “the” don’t carry scene semantics by themselves, but their frequent, consistent alignments signal similar syntactic frames (a prepositional phrase describing location).
  - Multiple `("is", "the")` pairs, each ≈ **0.0065**
  - Multiple `("a", "the")` pairs, each ≈ **0.0060**
  - `("with", "the")` pairs, each ≈ **0.0054**

  These function-word alignments (`a`, `the`, `is`, `with`, `in`) are less about “park vs playground” and more about saying: *the sentences have very similar grammatical structure and discourse style*, which also contributes to the match.

- Putting it together:
  - The **low energy (-0.423)** comes from many **positive token–token contributions**.
  - The **main semantic reasons** the model thinks they match are:
    - `("red","red")` → same color
    - `("ball","ball")` → same object
    - `("red","bright")` → compatible, reinforcing description of the color
  - The **supporting structural reasons** are the repeated high contributions from pairs like `("the","the")`, `("a","the")`, `("in","the")`, etc., which tell the model the sentences “look and feel” like descriptions of the same sort of scene.

These high-contribution pairs are exactly the **“this is why I think they match”** evidence in the joint-embedding energy view.

==============================================================
```
