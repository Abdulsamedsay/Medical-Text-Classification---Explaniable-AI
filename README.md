# XAI in Clinical Text Classification

## Comparing LIME and Integrated Gradients on Medical Transcriptions

**Course:** Explainable AI (SOW-BKI266)
**Institution:** Radboud University
**Academic Year:** 2025–2026

---

## Project Overview

In this project, I explore how two explainability methods, **LIME** and **Integrated Gradients**, help us understand the predictions of a transformer-based medical text classifier. The model is based on **DistilBERT** and is trained on clinical transcription data.

The main goal is to compare these two XAI methods in a medical NLP setting, with a focus on how understandable their explanations are and how well those explanations reflect the model’s actual decision-making process.

### Research Question

> How do LIME and Integrated Gradients compare in explaining the predictions of a transformer-based medical text classifier, in terms of faithfulness and interpretability?

---

## Repository Structure

```bash
xai-medical-text-classification/
├── xai_medical.ipynb                      # Main notebook with the full pipeline
├── README.md                              # Project documentation
├── class_distribution.png                 # Plot of class distribution
├── training_loss.png                      # Training loss over epochs
├── lime_explanation.png                   # Example output from LIME
├── integrated_gradients_explanation.png   # Example output from Integrated Gradients
├── comparison_lime_vs_ig.png              # Side-by-side comparison of both methods
└── faithfulness_deletion_test.png         # Plot for deletion-based faithfulness evaluation
```

---

## Requirements

To run this project, you will need:

* **Python 3.11**
* **VS Code** with the **Jupyter extension**, or any environment that supports Jupyter notebooks

You can install the dependencies by running the first notebook cell, or manually with:

```bash
pip install transformers torch captum lime scikit-learn pandas numpy matplotlib seaborn
```

---

## Dataset

This project uses the **Medical Transcriptions** dataset from Kaggle.

### How to get the dataset

1. Go to the dataset page on Kaggle:
   `https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions`
2. Download the file called `mtsamples.csv`
3. Place it in the same folder as `xai_medical.ipynb`

**Note:** The dataset is not included in this repository because of Kaggle’s terms of use.

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/[your-username]/xai-medical-text-classification.git
cd xai-medical-text-classification
```

2. Download `mtsamples.csv` from Kaggle and place it in the project folder.

3. Open `xai_medical.ipynb` in VS Code or Jupyter.

4. Run all cells from top to bottom.

The notebook includes the full workflow:

* loading and preprocessing the dataset
* selecting the top 5 medical specialties
* fine-tuning DistilBERT for classification
* generating LIME explanations
* generating Integrated Gradients explanations
* comparing both explanation methods visually
* evaluating faithfulness with a deletion test

---

## Results Summary

| Method               | Type                           | Interpretability             | Faithfulness                   |
| -------------------- | ------------------------------ | ---------------------------- | ------------------------------ |
| LIME                 | Model-agnostic, post-hoc       | High at word level           | Approximate                    |
| Integrated Gradients | Gradient-based, model-specific | Medium due to subword tokens | Stronger theoretical grounding |

### Model Performance

The DistilBERT classifier achieved an accuracy of **62%** on the test set for the **5-class classification task** after **3 training epochs**.

---

## XAI Methods Used

This project uses the following explainability techniques:

* **LIME**, implemented with the `lime` library using `LimeTextExplainer`
* **Integrated Gradients**, implemented with the `captum` library using `LayerIntegratedGradients`

---

## Reproducibility

To make the experiments reproducible, a fixed random seed is used throughout the project:

```python
SEED = 42
```
