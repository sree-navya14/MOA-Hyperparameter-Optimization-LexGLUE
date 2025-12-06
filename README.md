# 🚀 Parameter Optimization of Classical Machine Learning Classifiers using the Mayfly Optimization Algorithm (MOA) on the LexGLUE ECtHR-B Dataset

This project applies the **Mayfly Optimization Algorithm (MOA)** to optimize hyperparameters of classical machine learning classifiers for multi-label legal text classification on the **LexGLUE ECtHR-B** benchmark.  
The system integrates **Legal-BERT embeddings** with several classifiers, achieving significant performance improvements through metaheuristic optimization.

---

## 📌 Overview

Legal NLP tasks involve long documents, imbalanced labels, and domain-specific terminology.  
This project demonstrates that **optimized classical ML models**—combined with **dense transformer embeddings**—can still achieve strong performance on complex multi-label tasks.

The workflow includes:

1. Extracting 768-dimensional embeddings using **Legal-BERT**
2. Training classical classifiers (SVM, Logistic Regression, MLkNN, Random Forest, XGBoost)
3. Using **MOA** for hyperparameter tuning
4. Evaluating models using multi-label metrics such as F1-micro

---

## 🏗️ Architecture

Document Text
      ↓
Legal-BERT Embeddings (CLS Vector, 768d)
      ↓
Mayfly Optimization Algorithm (Hyperparameter Search)
      ↓
Optimized Classifiers (SVM, LR, MLkNN, RF, XGB)
      ↓
Evaluation (F1-micro, F1-macro, Hamming Loss)

---

## 🔍 Dataset: ECtHR Task B (LexGLUE)

- A multi-label legal judgement prediction dataset  
- Input: Long case descriptions from the European Court of Human Rights  
- Output: Violated ECHR articles (multi-label)  
- Highly imbalanced dataset, making optimization important  

**Dataset**
Download here: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#ECtHR%20(A)%20(LexGLUE)

---

## ⚙️ Features

- Legal-BERT embedding pipeline  
- Full Mayfly Optimization Algorithm implementation  
- Classifier-specific hyperparameter encoding  
- Multi-label evaluation metrics  
- Convergence plots for all classifiers  
- Reproducible training and evaluation pipeline  

---

## 📊 Results Summary

Final test performance of optimized models:

| Classifier | F1-Micro | F1-Macro | Hamming Loss |
|------------|----------|----------|---------------|
| **Logistic Regression** | **0.4053** | 0.0322 | 0.00805 |
| **SVM** | 0.3929 | 0.0323 | **0.00614** |
| **MLkNN** | 0.2783 | 0.0169 | 0.01099 |
| **XGBoost** | 0.2061 | 0.0072 | 0.00584 |

Highlights:

- Logistic Regression achieved the **highest overall score**
- SVM achieved the **lowest Hamming Loss**, giving the most stable performance
- MLkNN improved significantly after MOA tuning
- Tree-based model struggled with dense 768-dimensional embeddings

---

## 📦 Installation

```bash
git clone https://github.com/<sree-navya14>/MOA-Hyperparameter-Optimization-LexGLUE.git
cd MOA-Hyperparameter-Optimization-LexGLUE
```

## 🧠 Mayfly Optimization Algorithm — Short Explanation

MOA simulates mayfly behavior to optimize hyperparameters in a continuous search space.

- **Male mayflies** move based on personal best and global best solutions  
- **Female mayflies** move toward the nearest male  
- **Offspring** are generated using linear crossover between male–female pairs  
- **Fitness Function:** F1-micro score computed on the validation split  

MOA effectively balances **exploration** (global search) and **exploitation** (local refinement), making it more powerful than grid search or random search for complex multi-label problems.

---

## 📌 Evaluation Metrics

The following metrics were used to evaluate all classifiers:

- **F1-Micro (Primary Metric for LexGLUE)**  
- **F1-Macro**  
- **Hamming Loss**  
- **Convergence Curves** (performance vs. optimization iterations)  
- **Per-Label Precision, Recall, F1** for detailed analysis

  ## 📘 References

- Chalkidis, I., Fergadiotis, M., Tsarapatsanis, D., Aletras, N., & Androutsopoulos, I.  
  **LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.**  
  arXiv:2110.00976 (2021).  
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#ECtHR%20(A)%20(LexGLUE)

- Zervoudakis, K., & Tsafarakis, S.  
  **Mayfly Optimization Algorithm.**  
  Computers & Industrial Engineering, Elsevier.
  
- HuggingFace Model Card  
  **nlpaueb/legal-bert-base-uncased**  
  https://huggingface.co/nlpaueb/legal-bert-base-uncased



