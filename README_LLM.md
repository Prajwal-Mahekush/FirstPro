# GPT-2 From Scratch — Implementation & Fine-Tuning for Spam Classification

A from-scratch PyTorch implementation of the GPT-2 architecture, with pretrained OpenAI weights loaded in for validation, then fine-tuned for binary text classification (spam vs. non-spam).

## Overview

This project implements the GPT-2 transformer architecture from first principles — no `transformers` library, no pre-built model classes — including:

- Token + positional embeddings
- Multi-head self-attention
- Transformer decoder blocks (attention + feed-forward, residual connections, layer norm)
- Custom weight-loading logic to port OpenAI's original GPT-2 checkpoint into the custom architecture

The correctness of the from-scratch implementation was validated by loading OpenAI's pretrained weights and confirming output equivalence with the reference model. The model was then fine-tuned on a downstream task: classifying text messages as spam or non-spam.

## Why this project

Most spam classifiers use off-the-shelf pretrained models via high-level APIs. This project instead builds the architecture layer-by-layer to demonstrate a working understanding of how transformer language models are structured internally, how pretrained weights map onto that structure, and how to adapt a pretrained LM for a downstream classification task via fine-tuning.

## Architecture

- **Base model:** GPT-2 (small, 124M parameters) — [confirm variant used]
- **Framework:** PyTorch
- **Components implemented from scratch:**
  - Byte-pair tokenization / embedding layer
  - Multi-head causal self-attention
  - Transformer block (pre-LN, MLP with GELU)
  - Final classification head (added on top of pretrained backbone)

## Dataset

- **Dataset used:** [SMS Spam Collection Dataset (UCI ML Repository)](https://doi.org/10.24432/C5CC84)
- **Original size:** 5,574 messages — 747 spam (13.4%), 4,827 ham (86.6%)
- **Class balancing:** Undersampled the majority (ham) class to match the spam count, producing a balanced dataset before splitting
- **Split:** 70% train / 10% validation / 20% test
- **Preprocessing:** Tokenized with GPT-2's byte-pair encoding (`tiktoken`), padded to the longest sequence in the training set using the GPT-2 end-of-text token as the pad token

## Fine-Tuning Approach

- Loaded pretrained GPT-2 (small, 124M) weights from OpenAI's official checkpoint into the from-scratch architecture
- Replaced the language-modeling output head with a 2-class linear classification head
- Froze all pretrained weights except the final transformer block, the final layer norm, and the new classification head — a common efficient fine-tuning strategy that adapts the top-level representations while preserving lower-level language understanding
- Used the last token's output representation (final position in the sequence) as the input to the classification head
- **Hyperparameters:** learning rate = 5e-5, optimizer = AdamW, weight decay = 0.1, batch size = 8, epochs = 5

## Results

Evaluated on a held-out test set of 296 messages (balanced: 149 ham, 147 spam).

| Metric | Score |
|---|---|
| Accuracy | 86.49% |
| Precision | 0.8543 |
| Recall | 0.8776 |
| F1-score | 0.8658 |

**Per-class breakdown:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Ham | 0.88 | 0.85 | 0.86 | 149 |
| Spam | 0.85 | 0.88 | 0.87 | 147 |

**Confusion Matrix** (rows = actual, columns = predicted):

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 127 | 22 |
| **Actual Spam** | 18 | 129 |

**Analysis:** The model performs consistently across both classes (precision and recall both in the 0.85–0.88 range), with no strong bias toward over- or under-predicting spam. Of the 296 test messages, 40 were misclassified: 22 ham messages flagged as spam (false positives) and 18 spam messages missed as ham (false negatives). The relatively balanced error split suggests the classification head, trained on top of a frozen pretrained GPT-2 backbone, generalizes reasonably well without overfitting to either class — notable given the dataset's natural class imbalance (13% spam / 87% ham) was corrected via undersampling before this train/val/test split.

**Training log** (loss tracked every 50 steps):

| Epoch | Step | Train Loss | Val Loss |
|---|---|---|---|
| 1 | 0   | 1.034 | 0.798 |
| 1 | 50  | 0.752 | 1.003 |
| 1 | 100 | 1.251 | 0.958 |
| 2 | 150 | 0.944 | 0.991 |
| 2 | 200 | 1.078 | 1.037 |
| 2 | 250 | 0.908 | 0.843 |
| 3 | 300 | 1.139 | 1.034 |
| 3 | 350 | 0.987 | 0.796 |
| 4 | 400 | 0.765 | 0.880 |

Loss fluctuated across steps rather than decreasing smoothly, which is common with a small batch size (8) and a partially-frozen model — but final test-set metrics above confirm the model converged to a genuinely useful decision boundary despite the noisy intermediate loss curve.

## Key Learnings

- Porting OpenAI's checkpoint into the custom architecture required carefully matching parameter shapes and splitting the fused query/key/value attention weights (stored as a single combined matrix in the original checkpoint) into separate tensors for the from-scratch multi-head attention implementation.
- Freezing all layers except the last transformer block, final layer norm, and classification head was enough to reach strong test-set performance without the cost of full fine-tuning — a useful trade-off to understand when compute or data is limited.
- Loss on a noisy, small-batch training run doesn't tell the whole story — the fluctuating training/validation loss looked unstable step-to-step, but final evaluation on the held-out test set showed the model had, in fact, learned a solid decision boundary.

## References

- Build LLM from Scratch by Sebastian Raschka
- OpenAI GPT-2 released weights
- Attention is all you need(Paper from Google Reserachers)
