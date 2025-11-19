"""
jepa_ebm_demo_fixed.py

Demo of:
- OpenAI Responses API with GPT-5.1 function calling
- A custom tool that simulates a joint-embedding + energy-based model
  using local GloVe embeddings + OpenAI text embeddings.

Now extended so that the main energy is a pairwise interaction energy
over all GloVe token–token pairs:

    E(x, y) = - sum_{i=1}^m sum_{j=1}^n w_ij * phi(h_i, k_j)

where:
  - h_i, k_j are GloVe word vectors for context and candidate tokens,
  - phi is cosine similarity,
  - w_ij are softmax-normalized attention weights over all token pairs.

Requirements:
    pip install openai numpy

Assumptions about your filesystem:
    ~/Desktop/Glove/glove.6B.300d.txt
    or
    ~/Desktop/Glove/glove.840B.300d.txt
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# *** Replace this with your real key ***
OPENAI_API_KEY = "sk-proj-..."

# GPT model that supports tools/function calling
MODEL_NAME = "gpt-5.1"

# Embedding model used for the GPT side of the joint embedding
EMBEDDING_MODEL = "text-embedding-3-small"

# Candidate locations for your GloVe files (adapt as needed)
GLOVE_CANDIDATE_PATHS = [
    os.path.expanduser("~/Desktop/Glove/glove.6B.300d.txt"),
    os.path.expanduser("~/Desktop/Glove/glove.840B.300d.txt"),
]

# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------

if not OPENAI_API_KEY or "YOUR-OPENAI-API-KEY-HERE" in OPENAI_API_KEY:
    raise RuntimeError(
        "Please set OPENAI_API_KEY at the top of this file to your real OpenAI API key."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# GloVe loading & sentence embedding
# ---------------------------------------------------------------------

_glove_embeddings: Optional[Dict[str, np.ndarray]] = None
_glove_dim: Optional[int] = None
_glove_path: Optional[str] = None


def _find_glove_path() -> str:
    """Pick the first existing GloVe file from the candidate paths."""
    for path in GLOVE_CANDIDATE_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No GloVe file found. Update GLOVE_CANDIDATE_PATHS to point to "
        "glove.6B.300d.txt or glove.840B.300d.txt."
    )


def _load_glove_embeddings() -> None:
    """Load GloVe embeddings into memory (lazy, called on first use)."""
    global _glove_embeddings, _glove_dim, _glove_path

    if _glove_embeddings is not None:
        return

    _glove_path = _find_glove_path()
    print(f"[INFO] Loading GloVe embeddings from: {_glove_path}")
    embeddings: Dict[str, np.ndarray] = {}

    with open(_glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if not parts:
                continue
            word = parts[0]
            try:
                vec = np.asarray(parts[1:], dtype="float32")
            except ValueError:
                # Skip malformed lines
                continue
            embeddings[word] = vec

    if not embeddings:
        raise RuntimeError(f"Failed to load any GloVe vectors from {_glove_path}")

    # Infer dimensionality from first vector
    _glove_dim = len(next(iter(embeddings.values())))
    _glove_embeddings = embeddings
    print(f"[INFO] Loaded {len(_glove_embeddings)} GloVe vectors of dim {_glove_dim}.")


_token_pattern = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    """Very lightweight tokenizer: letters and apostrophes only, lowercased."""
    return [t.lower() for t in _token_pattern.findall(text)]


def glove_sentence_embedding(text: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute a simple sentence embedding by averaging GloVe word vectors.

    Returns:
        emb: sentence embedding (mean of token vectors or zero if no tokens)
        tokens_used: list of tokens that had in-vocab vectors
        token_vectors: array of shape (len(tokens_used), glove_dim)
    """
    _load_glove_embeddings()
    assert _glove_embeddings is not None
    assert _glove_dim is not None

    tokens = _tokenize(text)
    vectors: List[np.ndarray] = []
    used_tokens: List[str] = []

    for t in tokens:
        vec = _glove_embeddings.get(t)
        if vec is not None:
            vectors.append(vec)
            used_tokens.append(t)

    if not vectors:
        emb = np.zeros(_glove_dim, dtype="float32")
        token_vectors = np.zeros((0, _glove_dim), dtype="float32")
    else:
        token_vectors = np.stack(vectors, axis=0).astype("float32")
        emb = np.mean(token_vectors, axis=0).astype("float32")

    return emb, used_tokens, token_vectors


# ---------------------------------------------------------------------
# GPT embedding helpers
# ---------------------------------------------------------------------

def gpt_embeddings_for_pair(context: str, candidate: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get OpenAI embeddings for (context, candidate) in a single API call."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[context, candidate],
    )
    ctx_vec = np.array(response.data[0].embedding, dtype="float32")
    cand_vec = np.array(response.data[1].embedding, dtype="float32")
    return ctx_vec, cand_vec


def _project_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Force an embedding to a given dimension by truncation or zero-padding.
    """
    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    # pad with zeros
    pad_width = target_dim - vec.shape[0]
    return np.pad(vec, (0, pad_width), mode="constant")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------
# Pairwise token–token energy helper
# ---------------------------------------------------------------------

def _pairwise_token_energy(
    ctx_token_vecs: np.ndarray,
    cand_token_vecs: np.ndarray,
    tokens_ctx: List[str],
    tokens_cand: List[str],
    temperature: float = 0.5,
    top_k: int = 20,
) -> Tuple[float, List[dict]]:
    """
    Compute pairwise interaction energy and top contributing token–token pairs.

    E(x, y) = - sum_{i,j} w_ij * phi(h_i, k_j)

    where:
        phi(h_i, k_j) = cosine similarity
        w_ij = softmax(phi(h_i, k_j) / temperature) over all (i, j)
    """
    m = ctx_token_vecs.shape[0]
    n = cand_token_vecs.shape[0]

    if m == 0 or n == 0:
        # No tokens => no pairwise structure
        return 0.0, []

    # Normalize token vectors
    ctx_norms = np.linalg.norm(ctx_token_vecs, axis=1, keepdims=True)
    ctx_norms[ctx_norms == 0.0] = 1.0
    ctx_norm = ctx_token_vecs / ctx_norms

    cand_norms = np.linalg.norm(cand_token_vecs, axis=1, keepdims=True)
    cand_norms[cand_norms == 0.0] = 1.0
    cand_norm = cand_token_vecs / cand_norms

    # phi(h_i, k_j) as cosine similarity matrix (m x n)
    pair_scores = ctx_norm @ cand_norm.T  # cosines

    # Attention weights w_ij via softmax over all pairs
    logits = pair_scores.flatten() / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    weights_flat = exp_logits / np.sum(exp_logits)
    weights = weights_flat.reshape(m, n)

    # Contributions and energy
    contributions = weights * pair_scores  # w_ij * phi(h_i, k_j)
    energy = -float(np.sum(contributions))  # E(x,y) = - sum w_ij * phi

    # Extract top-k contributions for interpretability
    contrib_flat = contributions.flatten()
    # large positive contributions are the "this is why they match" pairs
    top_k = min(top_k, m * n)
    if top_k <= 0:
        return energy, []

    top_indices = np.argsort(-contrib_flat)[:top_k]

    top_pairs: List[dict] = []
    for idx in top_indices:
        i = int(idx // n)
        j = int(idx % n)
        top_pairs.append(
            {
                "context_token": tokens_ctx[i],
                "candidate_token": tokens_cand[j],
                "phi_cosine": float(pair_scores[i, j]),
                "weight": float(weights[i, j]),
                "contribution": float(contributions[i, j]),
                "indices": {"i": i, "j": j},
            }
        )

    return energy, top_pairs


# ---------------------------------------------------------------------
# JEPA-style joint embedding + EBM energy (simulated)
# ---------------------------------------------------------------------

def evaluate_joint_embedding_energy(
    context: str,
    candidate: str,
    verbose: bool = False,
) -> dict:
    """
    Simulate a JEPA-style joint embedding + energy-based model.

    Steps:
      - Build GloVe token embeddings for context and candidate.
      - Build GloVe sentence embeddings by averaging token vectors.
      - Build GPT embeddings for context and candidate.
      - Project GPT embeddings down to the same dimensionality as GloVe.
      - Form "joint" sentence embeddings by concatenating normalized
        GloVe + normalized GPT sentence vectors.
      - Define two kinds of similarity / energy:

        (1) Sentence-level joint cosine:
            cos_joint = cos(joint_context, joint_candidate)
            legacy_energy_joint = 1 - cos_joint

        (2) Token-level pairwise energy (main energy):
            E(x, y) = - sum_{i,j} w_ij * phi(h_i, k_j)
            where phi is cosine between GloVe token vectors and
            w_ij are softmax-normalized attention weights over all pairs.

        The returned `energy` field is this pairwise energy (2).
        Lower energy => stronger aggregate token–token match.
    """
    _load_glove_embeddings()
    assert _glove_dim is not None

    # Local word-level embeddings (JEPA-style latent for text)
    glove_ctx, tokens_ctx, ctx_token_vecs = glove_sentence_embedding(context)
    glove_cand, tokens_cand, cand_token_vecs = glove_sentence_embedding(candidate)

    # Model-level sentence embeddings
    gpt_ctx, gpt_cand = gpt_embeddings_for_pair(context, candidate)

    # Match dimensionality
    gpt_ctx_proj = _project_to_dim(gpt_ctx, _glove_dim)
    gpt_cand_proj = _project_to_dim(gpt_cand, _glove_dim)

    # Build joint sentence embeddings (concatenate normalized parts)
    joint_ctx = np.concatenate(
        [_l2_normalize(glove_ctx), _l2_normalize(gpt_ctx_proj)]
    )
    joint_cand = np.concatenate(
        [_l2_normalize(glove_cand), _l2_normalize(gpt_cand_proj)]
    )

    # Sentence-level cosine similarities
    cos_glove = _cosine(glove_ctx, glove_cand)
    cos_gpt = _cosine(gpt_ctx_proj, gpt_cand_proj)
    cos_joint = _cosine(joint_ctx, joint_cand)

    # Legacy joint energy: low energy == high compatibility (kept for reference)
    legacy_joint_energy = 1.0 - cos_joint  # roughly in [0, 2]

    # Token-level pairwise energy + top contributing token pairs
    pairwise_energy, top_pairs = _pairwise_token_energy(
        ctx_token_vecs=ctx_token_vecs,
        cand_token_vecs=cand_token_vecs,
        tokens_ctx=tokens_ctx,
        tokens_cand=tokens_cand,
        temperature=0.5,
        top_k=20,
    )

    result = {
        "context": context,
        "candidate": candidate,
        # MAIN ENERGY: token-level pairwise interaction energy
        "energy": float(pairwise_energy),
        "energy_definition": (
            "E(x, y) = - sum_{i,j} w_ij * phi(h_i, k_j) over GloVe token embeddings, "
            "where phi is cosine similarity and w_ij are softmax-normalized attention "
            "weights over all token pairs. Lower energy means a stronger aggregate "
            "token–token match between context and candidate."
        ),
        # Sentence-level diagnostics (unchanged from earlier version)
        "joint_cosine_similarity": float(cos_joint),
        "glove_cosine_similarity": float(cos_glove),
        "gpt_cosine_similarity": float(cos_gpt),
        "legacy_joint_energy": float(legacy_joint_energy),
        # Tokenization info
        "glove_tokens_context": tokens_ctx,
        "glove_tokens_candidate": tokens_cand,
        "glove_dim": int(_glove_dim),
        "gpt_embedding_dim": int(gpt_ctx.shape[0]),
        # Interpretability: top token–token contributions
        "token_pair_contributions_topk": top_pairs,
        "interpretation": (
            "The main energy is now a sum over all pairwise interactions between "
            "GloVe token embeddings from the context and the candidate. Each "
            "token–token pair contributes approximately contrib_ij = w_ij * phi(h_i, k_j), "
            "where phi is cosine similarity and w_ij is an attention weight derived "
            "from a softmax over all pairs. The largest positive contributions indicate "
            "the token pairs that most strongly support the match between context and "
            "candidate — these are the 'this is why I think they match' pairs."
        ),
    }

    if verbose:
        # Include norms and shape diagnostics
        result["norms"] = {
            "glove_context_norm": float(np.linalg.norm(glove_ctx)),
            "glove_candidate_norm": float(np.linalg.norm(glove_cand)),
            "gpt_context_norm": float(np.linalg.norm(gpt_ctx_proj)),
            "gpt_candidate_norm": float(np.linalg.norm(gpt_cand_proj)),
        }
        result["token_matrix_shapes"] = {
            "context_tokens": int(ctx_token_vecs.shape[0]),
            "candidate_tokens": int(cand_token_vecs.shape[0]),
        }

    return result


# ---------------------------------------------------------------------
# Tool definition for GPT function calling (Responses API)
# ---------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "evaluate_joint_embedding_energy",
        "description": (
            "Simulate a joint-embedding + energy-based model over text. "
            "Given a context and a candidate string, build a joint embedding "
            "using local GloVe vectors and OpenAI embeddings, then return an "
            "energy score and token-level pairwise contributions. Lower energy "
            "means the candidate is more compatible with the context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": (
                        "Background or conditioning text (question, scene, premise, etc.)."
                    ),
                },
                "candidate": {
                    "type": "string",
                    "description": (
                        "Hypothesis / answer / continuation text to evaluate "
                        "against the given context."
                    ),
                },
                "verbose": {
                    "type": "boolean",
                    "description": "If true, return extra diagnostics.",
                    "default": False,
                },
            },
            # strict tools require required to list *all* properties
            "required": ["context", "candidate", "verbose"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

AVAILABLE_FUNCTIONS = {
    "evaluate_joint_embedding_energy": evaluate_joint_embedding_energy,
}


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_jepa_ebm_demo(context: str, candidate: str, verbose: bool = True) -> None:
    """
    Orchestrates:
      1. Ask GPT-5.1 to assess candidate vs context, with TOOLS enabled.
      2. If it emits a function_call, execute our local Python function.
      3. Send the tool result back to the model.
      4. Print the final natural-language explanation.

    The model is instructed to explain the pairwise energy and to highlight
    the largest token–token contributions contrib_ij ≈ w_ij * phi(h_i, k_j)
    as "this is why I think they match."
    """
    user_prompt = (
        "You are simulating a simple joint-embedding + energy-based model "
        "inspired by JEPA + EBM ideas.\n\n"
        "Use the `evaluate_joint_embedding_energy` function to score how "
        "compatible the candidate is with the context. The tool now returns "
        "a pairwise interaction energy E(x,y) = - sum_{i,j} w_ij * phi(h_i, k_j) "
        "over all GloVe token–token pairs, as well as the top contributing "
        "token pairs with their approximate contributions contrib_ij ≈ "
        "w_ij * phi(h_i, k_j).\n\n"
        "In your explanation, do the following:\n"
        "  - Explain the sign and magnitude of the overall energy (lower is better).\n"
        "  - Use the provided token_pair_contributions_topk entries to highlight "
        "    which specific token–token pairs most strongly support the match.\n"
        "  - Make clear that those high-contribution pairs are 'this is why I think "
        "    they match'.\n\n"
        f"Context: {context}\n"
        f"Candidate: {candidate}\n"
    )

    input_items = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # First call: model may decide to call the function
    response = client.responses.create(
        model=MODEL_NAME,
        input=input_items,
        tools=TOOLS,
    )

    first_item = response.output[0]
    tool_result_dict = None

    if first_item.type == "function_call":
        tool_call = first_item
        args = json.loads(tool_call.arguments)
        func_name = tool_call.name
        print(f"\n[DEBUG] Model requested tool `{func_name}` with args:")
        print(json.dumps(args, indent=2))

        func = AVAILABLE_FUNCTIONS.get(func_name)
        if func is None:
            raise RuntimeError(f"Unknown tool requested: {func_name}")

        # Execute the local Python function
        tool_result_dict = func(**args)
        if verbose:
            print("\n[DEBUG] Raw tool result:")
            print(json.dumps(tool_result_dict, indent=2))

        # Feed the function call + its output back into the Responses API
        input_items.append(tool_call)
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": json.dumps(tool_result_dict),
            }
        )

        # Second call: model now has tool output, so it can explain it
        response = client.responses.create(
            model=MODEL_NAME,
            input=input_items,
            tools=TOOLS,
        )

    print("\n================ JEPA + EBM MODEL EXPLANATION ================\n")
    print(response.output_text)
    print("\n==============================================================\n")

    if tool_result_dict is not None and not verbose:
        print("Energy value:", tool_result_dict.get("energy"))


# ---------------------------------------------------------------------
# Main entry point (no CLI args needed)
# ---------------------------------------------------------------------

def main() -> None:
    print("JEPA-style joint embedding + energy-based model demo.")
    print("Press Enter for defaults if you don't feel like typing.\n")

    default_context = "A child is playing with a red ball in the park."
    default_candidate = "The kid happily throws the bright red ball across the playground."

    context = input(f"Context   [{default_context!r}]: ").strip()
    if not context:
        context = default_context

    candidate = input(f"Candidate [{default_candidate!r}]: ").strip()
    if not candidate:
        candidate = default_candidate

    run_jepa_ebm_demo(context, candidate, verbose=True)


if __name__ == "__main__":
    main()
