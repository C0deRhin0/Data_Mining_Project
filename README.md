# IoT Network Intrusion Detection with Hybrid Feature Selection and Random Forest

A high-performance intrusion detection framework for IoT networks using a two-stage feature selection pipeline—hybrid filter-wrapper technique (mutual information + RFECV)—and a tuned Random Forest classifier. Built and tested on the UNSW-NB15 dataset for real-time threat mitigation.

## 📌 Overview

This project aims to address the challenge of high-dimensional IoT traffic in intrusion detection by:
- Reducing irrelevant and redundant features via a **hybrid feature selection** approach
- Training an optimized **Random Forest** classifier with **GridSearchCV**
- Testing model robustness under **noise**, **feature ablation**, and **alternative classifiers**

## 🚀 Features

- **Recursive Feature Elimination with Cross-Validation (RFECV)** + **Filter-Based Ranking**
- Dimensionality reduction by **31.0%** (42 → 29 features)
- Random Forest model with **94.91% accuracy** and **96.01% F1-score**
- Robust to 5% **Gaussian noise** and **missing values**
- Benchmarked against **SVM**, **XGBoost**, and **SelectKBest + RF**
- Built-in support for **feature ablation studies**, **cross-validation**, and **noise injection tests**

## 🧪 Dataset

- **UNSW-NB15** network traffic dataset
- Stratified 70/30 split
- Data cleaning includes:
  - Removal of 9 metadata columns (e.g. IPs, timestamps)
  - Median/mode imputation
  - Label encoding & MinMax normalization

## 🛠️ Stack

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-Learn**, **XGBoost**, **SciPy**
- **Jupyter Notebook** for visualization and exploration

## 🧮 Methodology

1. **Data Preprocessing**
   - Cleaning, encoding, normalization
2. **Feature Selection**
   - Mutual Information + Random Forest importance
   - RFECV to select final 29 features
3. **Model Training**
   - GridSearchCV tuning (trees, depth, class weights)
4. **Evaluation**
   - Accuracy, F1, confusion matrix, Cohen’s Kappa, cross-validation
5. **Robustness Tests**
   - Feature ablation and noise injection
6. **Alternate Models**
   - Benchmarked vs SVM, XGBoost, and SelectKBest+RF

## 📊 Results

| Metric        | Value      |
|---------------|------------|
| Accuracy      | 94.91%     |
| F1-Score      | 96.01%     |
| Precision     | 96.29%     |
| Recall        | 95.73%     |
| Cohen’s Kappa | 0.8900     |
| Train Time    | 6.58 sec   |
| Inference     | 0.085 µs   |

## 📁 Repository Structure

```bash
📦 IoT-IDS-RFECV
 ┣ 📜 datamaining.py             # Main Python script
 ┣ 📜 Latex_Documentation.pdf    # Project paper with methodology & results
 ┗ 📁 Datasets/
    ┣ 📜 UNSW_NB15_training-set.csv
    ┗ 📜 UNSW_NB15_testing-set.csv
````

## ⚙️ Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/IoT-IDS-RFECV.git
   cd IoT-IDS-RFECV
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python datamaining.py
   ```

## 📄 Documentation

For detailed explanation of methodology, experimental design, and results, refer to [`Latex_Documentation.pdf`](./Latex_Documentation.pdf).

## 🤝 Contribution

Feel free to fork the repository and contribute improvements, test it on other datasets, or integrate other classifiers.

1. Fork the repo
2. Create a feature branch:

   ```bash
   git checkout -b new-feature
   ```
3. Commit changes and open a PR

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

* [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
* [Scikit-learn](https://scikit-learn.org/)
* [XGBoost](https://xgboost.ai/)
* AdNU CS Department Capstone Team (2025)

---

> ⚠️ For educational and research purposes only. Do not use for unauthorized network testing.

```

Let me know if you want a downloadable `.md` version or a modified tone (academic, casual, executive summary, etc.).
```
