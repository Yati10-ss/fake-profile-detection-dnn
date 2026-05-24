# 🛡️ Robust Fake Profile Detection Using Deep Neural Networks with Adversarial Training

A deep learning framework for detecting fake Instagram profiles, enhanced with **adversarial training** to maintain robustness against FGSM, PGD, and Carlini-Wagner attacks — achieving **88% accuracy under all adversarial conditions** while a standard DNN drops to 83%.

> Developed as part of the Robust Deep Learning course, MS in Artificial Intelligence & Machine Learning — Drexel University (2025).

---

## 🎯 Problem Statement

Social media platforms face a growing threat from fake profiles facilitating misinformation, phishing, and identity theft. Traditional machine learning classifiers achieve high accuracy on clean data but are highly vulnerable to **adversarial attacks** — subtle, mathematically crafted perturbations designed to fool the model.

This project addresses that vulnerability by:
- Building a baseline DNN for fake Instagram profile detection
- Demonstrating its vulnerability under real adversarial attack scenarios
- Implementing **PGD-based adversarial training** to produce a robust model
- Evaluating both models under **three distinct attack types**

---

## 🏗️ System Architecture

```
Instagram Profile Features
         │
         ▼
┌─────────────────────────────────────────┐
│         Preprocessing Pipeline          │
│                                         │
│  Feature Selection (10 features)        │
│  SMOTE Oversampling (class balance)     │
│  StandardScaler Normalization           │
│  70/30 Stratified Train/Test Split      │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────────┐
│ Standard DNN│  │      Robust DNN      │
│             │  │                      │
│ Input: 64   │  │ Input: 64 neurons    │
│ Hidden: 32  │  │ Hidden: 32 neurons   │
│ Hidden: 32  │  │ Hidden: 32 neurons   │
│ Output: 2   │  │ Output: 2 neurons    │
│             │  │          +           │
│ Clean data  │  │ PGD adversarial      │
│ training    │  │ examples during      │
│ only        │  │ training             │
└──────┬──────┘  └──────────┬───────────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────────┐
│         Adversarial Evaluation       │
│                                      │
│  Clean Data  →  Standard vs Robust   │
│  FGSM Attack →  Standard vs Robust   │
│  PGD Attack  →  Standard vs Robust   │
│  CW Attack   →  Standard vs Robust   │
└──────────────────────────────────────┘
```

---

## 📊 Results

### Standard DNN Performance

| Scenario | Accuracy | Real Recall | Fake F1 |
|----------|----------|-------------|---------|
| Clean | **0.92** | 0.79 | 0.96 |
| FGSM | 0.88 | **0.00** ⚠️ | 0.94 |
| PGD | 0.88 | **0.00** ⚠️ | 0.94 |
| CW | 0.83 | **0.00** ⚠️ | 0.91 |

### Robust DNN Performance (PGD Adversarial Training)

| Scenario | Accuracy | Real Recall | Fake F1 |
|----------|----------|-------------|---------|
| Clean | 0.91 | 0.82 | 0.95 |
| FGSM | **0.88** ✅ | **0.82** ✅ | 0.93 |
| PGD | **0.88** ✅ | **0.82** ✅ | 0.93 |
| CW | **0.88** ✅ | **0.82** ✅ | 0.93 |


### 📂 Detailed Metrics

Full per-class evaluation metrics for all scenarios are 
available as CSV files in the `metrics/` folder:

- `standard_dnn_metrics.csv` — Standard DNN performance 
   across Clean, FGSM, PGD, and CW scenarios
- `robust_dnn_metrics.csv` — Robust DNN performance 
   across all scenarios  
- `comparative_analysis_results.csv` — Side-by-side 
   comparison of both models
   

### Key Findings

- 🏆 **Robust DNN maintains 88% accuracy across ALL adversarial scenarios** — standard DNN drops to 83% under CW
- ⚠️ **Standard DNN completely fails to detect real profiles under attack** — recall drops to 0.00 under FGSM, PGD, and CW
- ✅ **Robust DNN maintains real profile recall of 0.82 throughout** — adversarial training preserves minority class detection
- 📉 **Trade-off is minimal** — robust model sacrifices only 1% clean accuracy (91% vs 92%) for dramatically improved attack resilience

---

## 🔬 Methodology

### Dataset
- **Source:** Instagram profile metadata (236 profiles: 28 real, 208 fake)
- **Features Used (10):** `edge_followed_by`, `edge_follow`, `username_length`, `username_has_number`, `full_name_has_number`, `full_name_length`, `is_private`, `is_joined_recently`, `is_business_account`, `has_external_url`
- **Class Imbalance:** Addressed via SMOTE oversampling
- **Split:** 70% train / 30% test with stratification

### Model Architecture
```
Input Layer:   64 neurons
Hidden Layer 1: 32 neurons, ReLU, Dropout 0.2
Hidden Layer 2: 32 neurons, ReLU, Dropout 0.3
Output Layer:   2 neurons, Softmax
Optimizer:     Adam (lr=0.001)
Loss:          Cross-entropy (class frequency weighted)
Epochs:        200
```

### Adversarial Attacks Implemented

| Attack | Type | Parameters |
|--------|------|-----------|
| **FGSM** | Fast Gradient Sign Method | ε = 0.05 |
| **PGD** | Projected Gradient Descent | ε = 0.05, α = 0.01, 10 steps |
| **CW** | Carlini-Wagner L2 | c = 1, 100 steps, lr = 0.02 |

### Adversarial Training (Robust DNN)
```
For each training batch:
  1. Generate PGD adversarial examples (ε=0.05, α=0.01, 10 steps)
  2. Compute clean loss on original examples
  3. Compute adversarial loss on perturbed examples
  4. Total loss = clean loss + adversarial loss
  5. Backpropagate and update weights
```

---

## 📁 Repository Structure

```
fake-profile-detection-dnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── fake_profile_detection_dnn.ipynb
│
├── data/
│   └── instagram_dataset.csv
│
├── results/
│   ├── confusion_matrix_clean_standard.png
│   ├── confusion_matrix_cw_standard.png
│   ├── confusion_matrix_clean_robust.png
│   ├── confusion_matrix_cw_robust.png
│   ├── confusion_matrix_fgsm_standard.png
│   ├── confusion_matrix_pgd_standard.png
│   ├── confusion_matrix_fgsm_robust.png
│   ├── confusion_matrix_pgd_robust.png
│   ├── accuracy_comparison_bar.png
│   ├── training_loss_curves.png
│   ├── precision_recall_standard.png
│   ├── precision_recall_robust.png
│   ├── boundary_consistency_comparison.png
│   └── certified_robustness_comparison.png
│
├── metrics/
│   ├── standard_dnn_metrics.csv
│   ├── robust_dnn_metrics.csv
│   └── comparative_analysis_results.csv
│
└── docs/
    └── final_report.pdf
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- GPU optional (CPU sufficient for this dataset size)

### Installation

```bash
# Clone the repository
git clone https://github.com/Yati10-ss/fake-profile-detection-dnn.git
cd fake-profile-detection-dnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Option 1: Jupyter locally
jupyter notebook notebooks/fake_profile_detection_dnn.ipynb

# Option 2: Google Colab (recommended)
# Upload notebook to colab.research.google.com
# Upload instagram_dataset.csv to Colab session storage
```

---

## 📦 Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📈 Key Learnings

- **Adversarial vulnerability is not obvious from clean accuracy** — the standard DNN achieves 92% on clean data, masking catastrophic failure under minimal perturbations
- **SMOTE oversampling is essential but insufficient** — class imbalance handling improves baseline performance but adversarial training is needed for attack resilience
- **PGD adversarial training generalizes across attack types** — training against PGD alone produces a model robust to FGSM and CW as well, showing the breadth of adversarial robustness
- **Minority class is hardest to protect** — real profiles (minority class) suffer most under attack; adversarial training specifically preserves recall for underrepresented classes
- **Clean-robust trade-off is smaller than expected** — only 1% accuracy drop on clean data for dramatically improved robustness confirms adversarial training is worth the cost

---

## ⚠️ Limitations

- **Small dataset (236 samples)** — overfitting risk is high, particularly for the robust model
- **Computational overhead** — adversarial training approximately doubles training time
- **Single platform** — results validated only on Instagram profile metadata; generalization to other platforms untested
- **Feature-based only** — no content analysis (images, text posts) which could improve detection

---

## 🔮 Future Work

- Expand dataset to thousands of profiles for better generalization
- Incorporate multi-attack training (FGSM + CW) during robust training
- Explore graph neural networks modeling social connection patterns
- Add content-based features (caption text, image analysis)
- Implement certified robustness bounds for guaranteed performance

---

## 👥 Contributors

| Contributor | Role & Contributions |
|-------------|----------------------|
| **Yateen Sakhare** | Adversarial attack implementation (FGSM, PGD, CW), robustness evaluation pipeline, loss analysis, hyperparameter tuning |
| **Shweta Sharma** | Dataset preprocessing, DNN architecture design, performance metric computation, hyperparameter tuning |

---

## 📚 References

1. Guna Sherar et al. — Fake Profile Detection Using Deep Learning Algorithm. IRJET, 2024.
2. Chongyang Zhao et al. — Adversarial Example Detection for Deep Neural Networks: A Review. IEEE DSC, 2023.
3. Eben Charles & Ponnarasan Krishnan — Adversarial Attacks in Deep Learning. Feb 2024.
4. Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR, 2018.
5. Carlini & Wagner — Evaluating the Robustness of Neural Networks. IEEE S&P, 2017.

---

## 🏫 Academic Context

> Developed for the **Robust Deep Learning** course, MS Artificial Intelligence & Machine Learning program, **Drexel University** (2025).
