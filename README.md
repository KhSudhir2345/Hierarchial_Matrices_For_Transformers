# Hierarchical Matrices for Transformers

This repository contains the code to reproduce the results from our work:

> **Hierarchical Matrix Approximations in Transformers: From Vision to Language Models**
> Kh Sudhir, Ch Lithin Sai Kumar, Vaishnavi Gujjula — IIIT Bangalore

The full write-up is available as [`PE_Hmat.pdf`](./PE_Hmat.pdf).

---

## Overview

We investigate replacing dense weight matrices in transformer architectures with **Hierarchical matrices (H-matrices)**, which exploit low-rank structure in off-diagonal blocks to reduce parameter counts. Experiments span two settings:

1. **Vision Transformers (ViT)** — H-matrix QKV projections trained from scratch on CIFAR-10
2. **LLM Fine-tuning** — Additive H-matrix adapters (analogous to LoRA) applied to TinyLlama-1.1B across general, medical, and legal NLP benchmarks

---

## Repository Structure

| Folder | Task | Dataset | Approach |
|--------|------|---------|----------|
| `ViT/` | Image classification | CIFAR-10 | H-matrix QKV projections, trained from scratch |
| `WikiText/` | Causal language modelling | WikiText-2 | Direct weight replacement ($W \leftarrow \mathcal{H}$) |
| `MEDQA/` | Medical question answering | MedQA-USMLE, PubMedQA | Additive H-matrix adapters ($W_0 + \Delta W_\mathcal{H}$) |
| `CaseHOLD/` | Legal question answering | CaseHOLD (LexGLUE) | Additive H-matrix adapters ($W_0 + \Delta W_\mathcal{H}$) |

---

## Key Results

**ViT (CIFAR-10, N = 4096)**
- 48% reduction in parameters
- 15% reduction in training time per epoch
- ~2% drop in accuracy vs. dense baseline

**LLM Fine-tuning (TinyLlama-1.1B)**

| Task | H-MAT | LoRA |
|------|-------|------|
| MedQA (3 epochs) | 26.24% | 26.16% |
| PubMedQA (3 epochs) | 63.40% | 63.80% |
| CaseHOLD (3 epochs) | 78.7% | 80.2% |

H-MAT consistently closes to within 1–2% of LoRA at matched parameter counts across both specialised domains.

---

## Base Model

All LLM experiments use **TinyLlama-1.1B** (`TinyLlama-1.1B-intermediate-step-1431k-3T`) as the base model, available on HuggingFace.

---

*Part of the code in this repository was developed with assistance from AI tools (GitHub Copilot, Claude).*
