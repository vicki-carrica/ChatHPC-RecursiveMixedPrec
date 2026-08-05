# Agentic AI Augmentation for Recursive Mixed-Precision Linear Solvers

This repository contains the software artifacts, training datasets, and In-Context Learning (ICL) prompts for the NextLA recursive mixed-precision framework and Agentic AI augmentation experiments. 

The repository includes the ChatHPC Library (v25.7.1), PEFT fine-tuning configurations for StarCoder2-15B, custom mixed-precision training data (`C2_NextLA_Dataset`), and prompt hierarchies for evaluating recursive linear algebra solvers (LU and QR).

See the [ChatHPC App README](C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/README.md) for detailed technical specifications on using the underlying ChatHPC Library CLI application.

---

## Dependencies

### Software
This repository's scripts depend on [`uv`](https://docs.astral.sh/uv/) to manage the Python virtual environment and ensure all required dependencies are installed reproducibly. A full list of Python dependencies can be found in `C1_ChatHPC_Lib/ChatHPC-app-v25.7.1/pyproject.toml`.

Please install `uv` prior to running the setup scripts by following the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). This software stack was developed and tested on Ubuntu 22.04 LTS and is compatible with modern Linux distributions.

### Hardware
Fine-tuning and inference routines require modern CUDA-capable GPUs (e.g., NVIDIA Ampere A100, Hopper H100/H200). The workflows rely on standard PyTorch and Hugging Face `transformers` / `peft` libraries.

---

## Directory Structure

```txt
ChatHPC-RecursiveMixedPrec
├── 1_setup.sh             — Sets up the Python virtual environment and downloads the StarCoder2-15B base model.
├── 2_train.sh             — Executes parameter-efficient fine-tuning (PEFT/LoRA) using the NextLA dataset.
├── basemodels             — Storage directory for pretrained base models (e.g., StarCoder2-15B).
├── C1_ChatHPC_Lib         — ChatHPC Library CLI application (v25.7.1).
│   └── ChatHPC-app-v25.7.1
├── C2_NextLA_Dataset      — Fine-tuning dataset artifact.
│   └── mixed_prec_training.json — 280 prompt-context-answer tuples across 8 framework modules.
├── LU_ICL_Prompts         — In-Context Learning prompts for recursive LU factorization (Levels 1–5).
│   ├── Level1.txt
│   ├── Level2.txt
│   ├── Level3.txt
│   ├── Level4.txt
│   └── Level5.txt
├── QR_ICL_Prompts         — In-Context Learning prompts for recursive QR factorization (Levels 1–3).
│   ├── Level1.txt
│   ├── Level2.txt
│   └── Level3.txt
├── output                 — Target directory for training checkpoints and fine-tuned adapters.
├── config_initial.json    — Initial fine-tuning configuration.
├── config_refinement.json — Refinement fine-tuning configuration (StarCoder2, 2048 context window).
└── prompt_template.txt    — System prompt template used during fine-tuning and inference.
```

## Workflow Steps

### 1. Setup Environment and Base Model
Run the setup script to initialize the virtual environment via uv and download the pretrained StarCoder2-15B base model into the `basemodels/` directory:

bash 1_setup.sh

### 2. Execute Fine-Tuning
Train the parameter-efficient adapter on the `C2_NextLA_Dataset/mixed_prec_training.json` dataset:

bash 2_train.sh

## Evaluating In-Context Learning (ICL) Prompts
To evaluate frontier models on recursive linear solvers using In-Context Learning, submit the raw prompt texts directly to the target LLM. The prompts are structured hierarchically to test the model's ability to extrapolate implementation details:

* **LU Factorization (`LU_ICL_Prompts/`):** Contains Levels 1 through 5. Level 1 provides a zero-shot baseline with no framework context, while Level 5 provides the complete data structure context alongside explicit mathematical block formulations to guide the nested recursive generation.
* **QR Factorization (`QR_ICL_Prompts/`):** Contains Levels 1 through 3. These prompts are designed to test if the model can generalize the framework to structurally distinct algorithms that rely on two-column panel partitions rather than four-quadrant subdivisions.
