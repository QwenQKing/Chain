# CHAIN: Calibrated LLM Forecasting via Causal-Temporal Hypergraph Inference

### CHAIN: Calibrated LLM Forecasting via Causal-Temporal Hypergraph Inference

[📄 Paper](https://arxiv.org/abs/2609.36689) | [🌐 Homepage](https://qwenqking.github.io/Chain-Homepage/) | [🚀 Quick Start](#quick-start) | 💬 Contact (wenjinliu23@outlook.com)

---

## Overview

<div align="center">
  <img src="figs/fig1.png" width="80%"/>
</div>

**CHAIN** addresses calibration bias in probabilistic LLM forecasting by modeling the forecasting process over **causal-temporal hypergraphs** rather than relying only on post-hoc probability correction.

CHAIN decomposes forecasting into three stages: **evidence weighting**, **evidence aggregation**, and **source fusion**. It introduces the **Causal-Temporal Validity Function (CTVF)**, **Direction-Aware Noisy-OR**, and **causal-coverage-balanced adaptive fusion** to improve the reliability of probabilistic predictions.

<div align="center">
  <img src="figs/fig3.png" width="95%"/>
</div>

---

## Experimental Results

<div align="center">
  <img src="figs/fig4.png" width="100%"/>
</div>

Across eight forecasting benchmarks and multiple LLM backbones, **CHAIN consistently improves probability calibration while maintaining or improving forecasting accuracy**. The figure above shows its OOD performance on DeepSeek-V3.

---

## CHAIN Implementation

### Install Environment

```bash
conda create -n chain python=3.10 -y
conda activate chain
pip install -r requirements.txt
```

### Dataset Preparation

The repository already contains the processed forecasting data used by CHAIN:

```text
datasets/
├── knowledge/   # knowledge data for hypergraph construction
└── eval/        # evaluation data
```

### API Configuration

For the default OpenAI-compatible setup:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

If the forecasting backbone uses a separate OpenAI-compatible endpoint, set:

```bash
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="YOUR_BASE_URL"
```

### Quick Start

Build the causal-temporal hypergraphs:

```bash
bash build_all_kg.sh
```

Run CHAIN with GPT-4o-mini:

```bash
MODELS="gpt-4o-mini" bash run_chain_eval.sh
```

### Evaluation

Run a specific backbone with the corresponding API configuration:

```bash
MODELS="gpt-4o-mini" bash run_chain_eval.sh
MODELS="deepseek-v3" bash run_chain_eval.sh
MODELS="gemini-2.0-flash" bash run_chain_eval.sh
```

Evaluation logs are saved under `logs/`.

---

## BibTeX

If you find this work helpful for your research, please cite:

```bibtex
@misc{liu2026chain,
      title={CHAIN: Calibrated LLM Forecasting via Causal-Temporal Hypergraph Inference},
      author={Wenjin Liu and Chenxi Wang and Yue Lu and Zhe Cui and Haoran Luo},
      year={2026},
      eprint={2609.36689},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.36689}
}
```

For further questions, please contact: wenjinliu23@outlook.com.
