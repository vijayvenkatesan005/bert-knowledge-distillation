# BERT-to-DistilBERT Knowledge Distillation

This repository contains an end-to-end model compression pipeline implementing task-specific **Knowledge Distillation (KD)** to transfer predictive intelligence from a large teacher model (`bert-base-uncased`) to a highly optimized, compact student model (`distilbert-base-uncased`). 

The architecture is explicitly optimized for sentence-level semantic similarity and natural language inference workloads, utilizing standard evaluation tasks from the GLUE benchmark.

## 📌 Core Engineering Focus
Large transformer models offer state-of-the-art accuracy but suffer from intensive computation footprints, high hosting costs, and latency bottlenecks in production. This project addresses those exact constraints by compressing model volume while strictly minimizing accuracy degradation.

### Key Optimization Metrics
* **Latency Reduction:** Achieves up to **60% faster inference** speeds compared to the base teacher model.
* **Storage Footprint:** Compresses total model size by **40%** (reducing parameter count by roughly 44M parameters).
* **Accuracy Retention:** Retains over **95-97%** of the original teacher model's capability across specialized downstream text classification evaluations.

## 🛠️ Tech Stack & Ecosystem
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Core Tooling:** Hugging Face Ecosystem (`transformers`, `datasets`, `accelerate`), PyTorch Neural Network modules (`torch.nn`)

## 📂 Implementation Architecture
The codebase isolates task-specific execution workflows for distinct NLP evaluation paradigms:

* **`MNLI_training_script.py` (Natural Language Inference):** Manishes a 3-way classification task (entailment, contradiction, neutral) mapping relationship mechanics between a premise and hypothesis sentence.
* **`QQP_training_script.py` (Semantic Similarity):** Manages binary sentence-pair classification to programmatically determine duplicate query pairs using soft probability distributions.
* **`Venkatesan_Vijay_Knowledge_Distillation_Slides.pdf`:** Architectural deck breaking down optimization equations, soft target evaluations, loss convergence properties, and performance benchmarks.

## 🧠 Algorithmic Mechanics
Rather than training a student network solely on hard categorical ground-truth targets ($0$ or $1$), this pipeline incorporates a specialized **Knowledge Distillation Loss Function** based on KL-Divergence.

### Soft Targets & Temperature Scaling
The pipeline passes the raw output logits of both the teacher and student networks through a scaled Softmax activation function using a hyperparameter **Temperature ($T$)**:

$$\sigma(z_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

* Higher temperature levels ($T > 1$) soften the probability distribution, forcing the student model to learn the "dark knowledge" or relative similarities hidden within the teacher's incorrect predictions (e.g., recognizing that a picture of a cat shares structural properties with a dog compared to a car).

### Combined Loss Alignment
The final backpropagation pass minimizes a weighted combination of standard Cross-Entropy loss ($L_{CE}$) alongside Kullback-Leibler Divergence loss ($L_{KD}$):

$$\text{Total Loss} = \alpha \cdot L_{CE} + (1 - \alpha) \cdot T^2 \cdot L_{KD}$$

This dual-loss configuration ensures the student captures absolute task labels while mirroring the underlying decision-boundary nuances of the larger network.
