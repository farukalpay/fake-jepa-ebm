"""
jepa_ebm_demo_fixed.py

Demo of:
- OpenAI Responses API with GPT-5.1 function calling
- A custom tool that simulates a joint-embedding + energy-based model
  using local GloVe embeddings + OpenAI text embeddings.

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


def glove_sentence_embedding(text: str) -> Tuple[np.ndarray, List[str]]:
    """
    Compute a simple sentence embedding by averaging GloVe word vectors.

    Returns: (embedding, tokens_used)
    """
    _load_glove_embeddings()
    assert _glove_embeddings is not None
    assert _glove_dim is not None

    tokens = _tokenize(text)
    vectors = []
    used_tokens: List[str] = []

    for t in tokens:
        vec = _glove_embeddings.get(t)
        if vec is not None:
            vectors.append(vec)
            used_tokens.append(t)

    if not vectors:
        # No in-vocab tokens; return zero vector
        emb = np.zeros(_glove_dim, dtype="float32")
    else:
        emb = np.mean(np.stack(vectors, axis=0), axis=0).astype("float32")

    return emb, used_tokens


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
# JEPA-style joint embedding + EBM energy (simulated)
# ---------------------------------------------------------------------

def evaluate_joint_embedding_energy(
    context: str,
    candidate: str,
    verbose: bool = False,
) -> dict:
    """
    Simulate a JEPA-style joint embedding + energy-based model:

    - Build GloVe sentence embeddings for context and candidate.
    - Build GPT embeddings for context and candidate.
    - Project GPT embeddings down to the same dimensionality as GloVe.
    - Form "joint" embeddings by concatenating normalized GloVe + normalized GPT.
    - Define energy as: 1 - cosine(joint_context, joint_candidate).

      -> Lower energy  => context & candidate are more compatible
      -> Higher energy => they are less compatible

    Returns a JSON-serializable dict with diagnostics.
    """
    _load_glove_embeddings()
    assert _glove_dim is not None

    # Local word-level embeddings (JEPA-style latent for text)
    glove_ctx, tokens_ctx = glove_sentence_embedding(context)
    glove_cand, tokens_cand = glove_sentence_embedding(candidate)

    # Model-level embeddings
    gpt_ctx, gpt_cand = gpt_embeddings_for_pair(context, candidate)

    # Match dimensionality
    gpt_ctx_proj = _project_to_dim(gpt_ctx, _glove_dim)
    gpt_cand_proj = _project_to_dim(gpt_cand, _glove_dim)

    # Build joint embeddings (concatenate normalized parts)
    joint_ctx = np.concatenate(
        [_l2_normalize(glove_ctx), _l2_normalize(gpt_ctx_proj)]
    )
    joint_cand = np.concatenate(
        [_l2_normalize(glove_cand), _l2_normalize(gpt_cand_proj)]
    )

    # Cosine similarities
    cos_glove = _cosine(glove_ctx, glove_cand)
    cos_gpt = _cosine(gpt_ctx_proj, gpt_cand_proj)
    cos_joint = _cosine(joint_ctx, joint_cand)

    # Simulated energy: low energy == high compatibility
    energy = 1.0 - cos_joint  # roughly in [0, 2]

    result = {
        "context": context,
        "candidate": candidate,
        "energy": float(energy),
        "joint_cosine_similarity": float(cos_joint),
        "glove_cosine_similarity": float(cos_glove),
        "gpt_cosine_similarity": float(cos_gpt),
        "glove_tokens_context": tokens_ctx,
        "glove_tokens_candidate": tokens_cand,
        "glove_dim": int(_glove_dim),
        "gpt_embedding_dim": int(gpt_ctx.shape[0]),
        "interpretation": (
            "Lower energy means the candidate is more compatible with the context "
            "under this joint-embedding toy model. This is a simulation of a "
            "JEPA + energy-based objective; no training is performed."
        ),
    }

    if verbose:
        # Include norms for extra diagnostics
        result["norms"] = {
            "glove_context_norm": float(np.linalg.norm(glove_ctx)),
            "glove_candidate_norm": float(np.linalg.norm(glove_cand)),
            "gpt_context_norm": float(np.linalg.norm(gpt_ctx_proj)),
            "gpt_candidate_norm": float(np.linalg.norm(gpt_cand_proj)),
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
            "energy score. Lower energy means the candidate is more compatible "
            "with the context."
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
            # *** FIX: strict tools require required to list *all* properties ***
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
    """
    user_prompt = (
        "You are simulating a simple joint-embedding + energy-based model "
        "inspired by JEPA + EBM ideas. "
        "Use the `evaluate_joint_embedding_energy` function to score how "
        "compatible the candidate is with the context, then explain the "
        "energy value and what it means.\n\n"
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
