# STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?

[![arXiv](https://img.shields.io/badge/arXiv-2503.23765-b31b1b.svg)](https://arxiv.org/abs/2503.23765) [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/MIRA-SJTU/STI-Bench) [![GitHub Repo](https://img.shields.io/badge/GitHub-Code-lightgrey)](https://github.com/MIRA-SJTU/STI-Bench)

---

## Overview

The use of Multimodal Large Language Models (MLLMs) as an end-to-end solution for Embodied AI and Autonomous Driving is a growing trend. However, while MLLMs excel at semantic understanding, their ability to perform precise, quantitative spatial-temporal reasoning in real-world applications remains largely unexamined. To address this gap, we introduce the Spatial-Temporal Intelligence Benchmark (**STI-Bench**), detailed in our paper [*“STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?”*](https://arxiv.org/abs/2503.23765). STI-Bench evaluates MLLMs' spatial-temporal intelligence through challenging tasks on real-world video data, including estimating and predicting object appearance, pose, displacement, and motion. Our benchmark covers diverse robot and vehicle operations across desktop, indoor, and outdoor scenarios. Extensive experiments reveal that even state-of-the-art MLLMs struggle significantly with these tasks, particularly those requiring precise distance estimation and motion analysis, highlighting a critical area for future research and development.

![Cover Image](assets/images/cover.jpg)

---

## Results

<img src="assets/images/results.jpg" alt="Main Results Table" style="zoom: 2.55%;" /> <img src="assets/images/radar.jpg" alt="Radar Chart Results" style="zoom: 30%;" />

---
## RUN Your Own Evaluation

```python
from datasets import load_dataset
sti_bench = load_dataset("MIRA-SJTU/STI-Bench")
```
or you can:

```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/MIRA-SJTU/STI-Bench
```

Test code for openai api and open source models like Qwen 2.5 VL are provided in this repo.

---

## Conclusion

STI-Bench provides a comprehensive benchmark for evaluating MLLMs' spatial-temporal understanding. Our findings reveal significant limitations in current models, particularly in precise quantitative tasks, highlighting inaccuracies in spatial quantification, temporal dynamics understanding, and cross-modal integration. There is a substantial gap between current capabilities and the reliability needed for real-world applications like embodied AI and autonomous driving. STI-Bench serves as a valuable tool for driving progress in developing MLLMs that can accurately perceive and reason about the physical world.
