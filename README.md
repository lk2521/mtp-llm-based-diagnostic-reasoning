# Evaluating and Enhancing LLM-Based Diagnostic Reasoning

This repository contains the code, data, and resources for my Master of Technology thesis project submitted to the Indian Institute of Technology Kharagpur. The project investigates the diagnostic reasoning capabilities of open-source medical Large Language Models (LLMs) and explores how to improve them through targeted parameter-efficient finetuning (PEFT) and layer-wise analysis.

## 📌 Project Overview
* **Base Model:** GPT-OSS-20B, a 21-billion parameter Mixture-of-Experts (MoE) model.
* **Targeted Finetuning:** Instead of uniform full-model adaptation, this project selectively finetunes specific transformer components (Value and Output projections) using QLoRA.
* **Layer-Wise Analysis:** Explores restricting parameter updates to specific layer depths (e.g., Top-8, Top-4) to optimize reasoning performance and computational efficiency.
* **Datasets:** Models are trained using the `medical-o1-reasoning-SFT` dataset with Chain-of-Thought (CoT) supervision and evaluated on the clinical `OpenX-DDx` dataset.

## 📂 Repository Structure
Based on the project's development workflow, the repository includes:

* `knockout_base.py`: Implements a projection-level sensitivity analysis (layer knockout) to identify critical architectural components prior to finetuning.
* `finetune_experiment.py`: The main script for executing targeted LoRA finetuning on selected model components (e.g., v+o, v-only, MLP).
* `finetune_experiment_layer_wise.py`: Conducts layer-restricted finetuning configurations to evaluate the impact of adaptation depth.
* `evaluate_experiment.py`: Calculates semantic reasoning alignment using metrics such as BERTScore and embedding-based similarity.
* `test_openddx.py` / `test_openddx_notebook.ipynb`: Executes the downstream structured diagnostic assessment via threshold-based semantic matching.
* `Images & Plots/`: Contains training loss curves, category-wise F1 comparisons, and performance visualizations across specialties.
* `Results/`: Stores the output predictions and evaluation metrics generated during testing.

## ⚙️ Key Findings
1.  **Component Sensitivity:** Attention-based finetuning—specifically targeting the Value + Output (v+o) projections—significantly outperforms uniform or MLP-based adaptation for clinical diagnostic reasoning.
2.  **Upper-Layer Dominance:** Finetuning only the upper layers (Top-8) yields diagnostic accuracy that is comparable to, or better than, adapting the entire model footprint.
3.  **Evaluation Limitations:** Standard validation loss is an insufficient proxy for clinical reasoning quality; semantic evaluation methods provide a much more reliable signal for true model alignment.

## 🧑‍💻 Author
**Lavesh Devidas Kadam** (21MT3A136)  
Department of Artificial Intelligence, IIT Kharagpur  
*Supervised by Professor Jiaul Paik*
