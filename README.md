<div align="center">

<h1>
    `Training Prompt Matters: State-Adaptive Optimization for Robust Fine-Tuning`
    <br><br>
    <b>ICML 2026</b>
    <br><br>
    <a href="https://arxiv.org/abs/ARXIV_ID" target="_blank">
      <img src="https://img.shields.io/badge/Paper%20ArXiv-SAPO-b31b1b.svg" alt="Paper ArXiv: GRACE">
    </a>
  </h1>
</div>


# State-Adaptive Prompt Optimization (SAPO)

This repository contains the official implementation for our work **"Training Prompt Matters: State-Adaptive Optimization for Robust Fine-Tuning"**.

## 💡Introduction

Unlike previous prompt engineering works that solely focus on evaluation performance during inference, our work is the first to systematically investigate the impact of training prompts on model capabilities after fine-tuning<sup></sup>. We discover that:

1. While different paraphrased prompts have a negligible impact on the current task's performance, they exert a profound influence on catastrophic forgetting of previously trained tasks and generalization to unseen tasks<sup></sup>.
2. These cross-task impacts are positively correlated, indicating that training prompt formulation is a tractable optimization objective<sup></sup>.
3. There exists an effective pre-indicator (pre-update loss) to reliably identify superior prompts prior to learning<sup></sup>.

Based on these findings, we propose ​SAPO (State-Adaptive Prompt Optimization)​, a lightweight and dynamic training strategy<sup></sup>.

## 🛠 Setup

### 1. Environment

You can install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

### 2. API Configuration for Prompt Generation

Before training with SAPO, you need to configure the LLM used for generating paraphrase prompts.

Please modify the `API_KEY` and `URL` in the following file: ```src/prompt_gen/API.py```

## 📂 Data Preparation

The datasets used in our experiments are organized in the following directories:

* **SUPERNI**: Located in `SUPERNI/`
* **TRACE**: Located in `TRACE/`

## 🚀 Training

We provide training scripts for 4 methods: **LoRAInc**, **LoRAInc + SAPO**, **O-LoRA**, and **O-LoRA + SAPO**. The scripts are located in the `scripts/` folder.

Below are the examples for training **Qwen3-8B** on the **NI_SEQ_C1** task sequence.

### 1. LoRAInc

```bash
mkdir -p saves/LoraInc/qwen3-8b/ni_seq_c1/logs/ && \
bash scripts/LoraInc/qwen3-8b/ni_seq_c1.sh > saves/LoraInc/qwen3-8b/ni_seq_c1/logs/train.log 2>&1
```

### 2. LoRAInc + SAPO (Ours)

```bash
mkdir -p saves/LoraInc_SAPO/qwen3-8b/ni_seq_c1/logs/ && \
bash scripts/LoraInc_SAPO/qwen3-8b/ni_seq_c1.sh > saves/LoraInc_SAPO/qwen3-8b/ni_seq_c1/logs/train.log 2>&1
```

### 3. O-LoRA

```bash
mkdir -p saves/O-LoRA/qwen3-8b/ni_seq_c1/logs/ && \
bash scripts/O-LoRA/qwen3-8b/ni_seq_c1.sh > saves/O-LoRA/qwen3-8b/ni_seq_c1/logs/train.log 2>&1
```

### 4. O-LoRA + SAPO (Ours)

```bash
mkdir -p saves/O-LoRA_SAPO/qwen3-8b/ni_seq_c1/logs/ && \
bash scripts/O-LoRA_SAPO/qwen3-8b/ni_seq_c1.sh > saves/O-LoRA_SAPO/qwen3-8b/ni_seq_c1/logs/train.log 2>&1
```

## 📊 Evaluation

We provide the initial zero-shot performance of different models on SUPERNI and TRACE in the `original_performance/` directory.

To evaluate the Continual Learning metrics (AP, BWT, FWT), run `src/run_evaluation.py`.

**Example:** Evaluate LoRAInc on Qwen3-8B (NI_seq_C1):

```bash
python3 src/run_evaluation.py \
    --outputs_dir saves/LoraInc/qwen3-8b/ni_seq_c1/outputs \
    --initial_metric_file original_performance/qwen3-8b.json
```

**Arguments:**

* `--outputs_dir`: The directory containing the training outputs (e.g., subfolders like `1-taskXXX`, `2-taskXXX`).
* `--initial_metric_file`: The JSON file containing the model's initial performance.

## 📚 Citations

If you find this work useful, please kindly star the repo and and cite the paper below. For questions, contact wenhangshi@ruc.edu.cn, or open an issue. Thank you!

```bibtex
@misc{shi2025lossgaingatedrefinement,
      title={No Loss, No Gain: Gated Refinement and Adaptive Compression for Prompt Optimization}, 
      author={Wenhang Shi and Yiren Chen and Shuqing Bian and Xinyi Zhang and Kai Tang and Pengfei Hu and Zhe Zhao and Wei Lu and Xiaoyong Du},
      year={2025},
      eprint={2509.23387},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.23387}, 
}
```


## 💐 Acknowledgments

This repository benefits from [O-LoRA](https://github.com/cmnfriend/O-LoRA). We thank the authors for their wonderful work.