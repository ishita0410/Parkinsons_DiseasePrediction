# Parkinson's Disease Detection using Machine Learning

A machine learning project that detects Parkinson's Disease from biomedical voice measurements using multiple classification algorithms.

---

## About the Project

Parkinson's disease is a progressive neurodegenerative disorder affecting the neurons in the brain that produce dopamine. It impacts movement and causes tremors and stiffness, with an estimated **7–10 million people** affected worldwide. Currently, there is no cure.

This project uses voice signal data to classify whether a person has Parkinson's disease or is healthy, leveraging several ML models and comparing their performance.

---

## Dataset

- **Source:** Created by Max Little of the University of Oxford, in collaboration with the National Centre for Voice and Speech, Denver, Colorado
- **Instances:** 195
- **Attributes:** 23
- **Task:** Binary Classification (`1` = Parkinson's, `0` = Healthy)
- **Missing Values:** None

### Key Features

| Feature | Description |
|---|---|
| `MDVP:Fo(Hz)` | Average vocal fundamental frequency |
| `MDVP:Fhi(Hz)` | Maximum vocal fundamental frequency |
| `MDVP:Flo(Hz)` | Minimum vocal fundamental frequency |
| `MDVP:Jitter(%)`, `MDVP:RAP`, `MDVP:PPQ` | Variation in fundamental frequency |
| `MDVP:Shimmer`, `Shimmer:APQ3`, `MDVP:APQ` | Variation in amplitude |
| `NHR`, `HNR` | Noise-to-tonal component ratio |
| `RPDE`, `D2` | Nonlinear dynamical complexity measures |
| `DFA` | Signal fractal scaling exponent |
| `spread1`, `spread2`, `PPE` | Nonlinear measures of frequency variation |
| `status` | Target variable (1 = Parkinson's, 0 = Healthy) |

---

## Workflow

1. **Exploratory Data Analysis** — Distribution plots, boxplots for outlier detection
2. **Preprocessing** — Label encoding, handling class imbalance with `RandomOverSampler`, feature scaling with `MinMaxScaler`
3. **Dimensionality Reduction** — PCA retaining 95% variance
4. **Model Training** — Multiple classifiers trained and compared
5. **Evaluation** — Accuracy, confusion matrix, classification report, ROC curve, Precision-Recall curve

---

## Models Used

| Model | Library |
|---|---|
| Logistic Regression | `sklearn` |
| Decision Tree | `sklearn` |
| Random Forest (Gini) | `sklearn` |
| Random Forest (Entropy) | `sklearn` |
| Support Vector Machine (SVM) | `sklearn` |
| K-Nearest Neighbors (KNN) | `sklearn` |
| Gaussian Naive Bayes | `sklearn` |
| Bernoulli Naive Bayes | `sklearn` |
| Voting Classifier (Ensemble) | `sklearn` |
| XGBoost | `xgboost` |
| Neural Network | `tensorflow` / `keras` |

---

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve & AUC Score
- Precision-Recall Curve
- Cumulative Feature Importance
- Tree Depth Distribution (Random Forest)

---

## Libraries & Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow plotly
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Parkinsons_ML.git
   cd Parkinsons_ML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Parkinson_disease_using_machine_learning.ipynb
   ```

4. Make sure `parkinsons.data` is available in the correct path, or update the path in the notebook:
   ```python
   df = pd.read_csv('parkinsons.data')
   ```

---

## Project Structure

```
Parkinsons_ML/
│
├── Parkinson_disease_using_machine_learning.ipynb   # Main notebook
├── parkinsons.data                                  # Dataset
└── README.md                                        # Project documentation
```

---

## References

- [UCI ML Repository – Parkinsons Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [Fundamental Frequency Variation Paper](http://www.cs.cmu.edu/~kornel/pubs/003228.pdf)
- Max A. Little, et al. — "Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease"
