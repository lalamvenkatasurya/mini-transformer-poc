import re
import numpy as np


VOCAB = [
    "<PAD>",
    "<BOS>",
    "<UNK>",
    "ai",
    "learns",
    "patterns",
    "from",
    "data",
    "with",
    "attention",
    "models",
    "predict",
    "next",
    "tokens",
    ".",
]

TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}

ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}


def tokenize(text):
    pieces = re.findall(
        r"[A-Za-z]+|[.]",
        text.lower(),
    )

    if not pieces:
        return ["<UNK>"]

    return [piece if piece in TOKEN_TO_ID else "<UNK>" for piece in pieces]


def tokens_to_ids(tokens):
    return [TOKEN_TO_ID[token] for token in tokens]


D_MODEL = 8
D_FF = 16
SEED = 7

rng = np.random.default_rng(SEED)

EMBEDDINGS = rng.normal(
    0,
    0.5,
    size=(len(VOCAB), D_MODEL),
)

W_Q = rng.normal(
    0,
    1 / np.sqrt(D_MODEL),
    size=(D_MODEL, D_MODEL),
)

W_K = rng.normal(
    0,
    1 / np.sqrt(D_MODEL),
    size=(D_MODEL, D_MODEL),
)

W_V = rng.normal(
    0,
    1 / np.sqrt(D_MODEL),
    size=(D_MODEL, D_MODEL),
)

W_O = rng.normal(
    0,
    1 / np.sqrt(D_MODEL),
    size=(D_MODEL, D_MODEL),
)

W_1 = rng.normal(
    0,
    1 / np.sqrt(D_MODEL),
    size=(D_MODEL, D_FF),
)

B_1 = np.zeros(D_FF)

W_2 = rng.normal(
    0,
    1 / np.sqrt(D_FF),
    size=(D_FF, D_MODEL),
)

B_2 = np.zeros(D_MODEL)

UNEMBEDDING = EMBEDDINGS.T


def positional_encoding(
    length,
    d_model=D_MODEL,
):
    positions = np.arange(length)[:, np.newaxis]

    dimensions = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimensions // 2)) / d_model,
    )

    angles = positions * angle_rates

    encoding = np.zeros((length, d_model))

    encoding[:, 0::2] = np.sin(angles[:, 0::2])

    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding


def softmax(
    values,
    axis=-1,
):
    shifted = values - np.max(
        values,
        axis=axis,
        keepdims=True,
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=axis,
        keepdims=True,
    )


def layer_norm(
    x,
    epsilon=1e-5,
):
    mean = np.mean(
        x,
        axis=-1,
        keepdims=True,
    )

    variance = np.var(
        x,
        axis=-1,
        keepdims=True,
    )

    return (x - mean) / np.sqrt(variance + epsilon)


def transformer_forward(
    text,
    temperature=1.0,
):
    temperature = max(
        float(temperature),
        0.05,
    )

    tokens = tokenize(text)
    token_ids = tokens_to_ids(tokens)

    token_vectors = EMBEDDINGS[token_ids]

    positions = positional_encoding(len(tokens))

    x = token_vectors + positions

    q = x @ W_Q
    k = x @ W_K
    v = x @ W_V

    raw_scores = (q @ k.T) / np.sqrt(D_MODEL)

    causal_mask = np.triu(
        np.ones_like(
            raw_scores,
            dtype=bool,
        ),
        k=1,
    )

    masked_scores = raw_scores.copy()

    masked_scores[causal_mask] = -1e9
    attention_weights = softmax(
        masked_scores,
        axis=-1,
    )
    attention_output = attention_weights @ v

    attention_output = attention_output @ W_O
    after_attention = layer_norm(x + attention_output)
    hidden = np.maximum(
        0,
        after_attention @ W_1 + B_1,
    )

    ffn_output = hidden @ W_2 + B_2

    final_hidden = layer_norm(after_attention + ffn_output)
    logits = final_hidden[-1] @ UNEMBEDDING
    probabilities = softmax(logits / temperature)

    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "token_vectors": token_vectors,
        "positions": positions,
        "input_with_position": x,
        "q": q,
        "k": k,
        "v": v,
        "raw_scores": raw_scores,
        "masked_scores": masked_scores,
        "attention_weights": attention_weights,
        "attention_output": attention_output,
        "final_hidden": final_hidden,
        "logits": logits,
        "probabilities": probabilities,
    }


def top_predictions(
    probabilities,
    top_n=5,
):
    indices = np.argsort(probabilities)[-top_n:][::-1]

    return [
        (
            ID_TO_TOKEN[int(index)],
            float(probabilities[index]),
        )
        for index in indices
    ]


def sample_next_token(
    probabilities,
    seed=42,
):
    sample_rng = np.random.default_rng(seed)

    token_id = int(
        sample_rng.choice(
            len(VOCAB),
            p=probabilities,
        )
    )

    return ID_TO_TOKEN[token_id]


if __name__ == "__main__":
    print(tokenize("AI learns patterns from data."))
    print(tokenize("AI understands language."))

    prompt = "ai learns patterns from data"

    result = transformer_forward(
        prompt,
        temperature=1.0,
    )

    print(
        "Tokens:",
        result["tokens"],
    )

    print(
        "Token IDs:",
        result["token_ids"],
    )

    print(
        "Attention shape:",
        result["attention_weights"].shape,
    )

    print("\nAttention row sums:")

    print(
        np.round(
            result["attention_weights"].sum(axis=1),
            3,
        )
    )

    print("\nTop next-token probabilities:")

    for token, probability in top_predictions(
        result["probabilities"],
        top_n=5,
    ):
        print(
            token,
            round(
                probability,
                3,
            ),
        )

    print("\nSampled next token:")

    print(
        sample_next_token(
            result["probabilities"],
            seed=42,
        )
    )
