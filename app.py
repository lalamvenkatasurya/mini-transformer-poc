import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from transformer_core import (
    transformer_forward,
    top_predictions,
    sample_next_token,
)


st.set_page_config(
    page_title=("Transformer Forward-Pass Explorer"),
    layout="wide",
)

st.title("Transformer Forward-Pass Explorer")

st.write(
    "Inspect how a small "
    "Transformer-style block "
    "moves from text to "
    "next-token probabilities."
)

prompt = st.text_input(
    "Enter a prompt",
    value=("ai learns patterns from data"),
)

temperature = st.slider(
    "Temperature",
    min_value=0.2,
    max_value=2.0,
    value=1.0,
    step=0.1,
)
result = transformer_forward(
    prompt,
    temperature=temperature,
)

st.subheader("1. Tokenization")

token_table = pd.DataFrame(
    {
        "Token": result["tokens"],
        "Token ID": result["token_ids"],
    }
)

st.dataframe(
    token_table,
    use_container_width=True,
)
st.subheader("2. Numeric Representation")

st.write(
    "Token embedding shape:",
    result["token_vectors"].shape,
)

st.write(
    "Positional encoding shape:",
    result["positions"].shape,
)

st.write(
    "Combined input shape:",
    result["input_with_position"].shape,
)
with st.expander("Inspect the combined vectors"):
    st.dataframe(
        pd.DataFrame(
            result["input_with_position"],
            index=result["tokens"],
        ),
        use_container_width=True,
    )
st.subheader("3. Causal Self-Attention")

fig, ax = plt.subplots(figsize=(7, 6))

matrix = result["attention_weights"]

image = ax.imshow(matrix)

ax.set_xticks(range(len(result["tokens"])))

ax.set_xticklabels(
    result["tokens"],
    rotation=45,
    ha="right",
)

ax.set_yticks(range(len(result["tokens"])))

ax.set_yticklabels(result["tokens"])

ax.set_xlabel("Key position")

ax.set_ylabel("Query position")

ax.set_title("Attention Weights")

for row in range(len(result["tokens"])):
    for column in range(len(result["tokens"])):
        ax.text(
            column,
            row,
            f"{matrix[row, column]:.2f}",
            ha="center",
            va="center",
        )

fig.colorbar(
    image,
    ax=ax,
)

fig.tight_layout()

st.pyplot(fig)
st.caption(
    "Read one row at a time. "
    "The causal mask prevents "
    "a position from attending "
    "to future positions."
)
st.subheader("4. Next-Token Probabilities")

top = top_predictions(
    result["probabilities"],
    top_n=8,
)

probability_table = pd.DataFrame(
    top,
    columns=[
        "Token",
        "Probability",
    ],
)

st.dataframe(
    probability_table,
    use_container_width=True,
)

st.bar_chart(probability_table.set_index("Token"))
sampled_token = sample_next_token(
    result["probabilities"],
    seed=42,
)

st.metric(
    "Sampled next token",
    sampled_token,
)
st.info(
    "This project uses fixed "
    "random weights, so the "
    "sampled token is not "
    "expected to be meaningful "
    "language. The purpose is "
    "to inspect the Transformer "
    "forward-pass mechanics."
)
