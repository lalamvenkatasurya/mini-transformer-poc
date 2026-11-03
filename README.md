# Transformer Forward-Pass Explorer

## Problem

Transformer diagrams can make the architecture
look like a black box.

This project makes the forward pass inspectable.

## What I Built

- tiny tokenizer
- token embeddings
- positional encoding
- Q / K / V projections
- causal self-attention
- residual connection
- layer normalization
- feed-forward network
- vocabulary logits
- temperature
- next-token probability visualization
- interactive Streamlit dashboard

## Architecture

Text
→ Tokens
→ Embeddings + Position
→ Causal Self-Attention
→ Add & Norm
→ Feed Forward
→ Add & Norm
→ Logits
→ Temperature
→ Softmax
→ Next Token

## Key Learning

[Write this yourself.]

## Experiments

[Summarize your position, mask,
temperature, and architecture experiments.]

## Limitations

- tiny vocabulary
- random fixed weights
- one simplified attention head
- one simplified Transformer block
- no training
- not intended to generate useful language

## Next Steps

Keep this high-level.
Do not implement later course topics yet.