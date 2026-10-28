# AI-Based Synthetic Telecom Customer Data Generation and Churn Prediction Using GANs and Deep Learning

An end-to-end machine learning project that generates synthetic telecom customer data with CTGAN and predicts customer churn using classical and deep learning models, culminating in an FT-Transformer classifier and a simple prediction interface.

## Overview

Telecom churn datasets are typically imbalanced, with far fewer churners than non-churners. This project compares ways of handling that imbalance (class weights, SMOTE, and GAN-generated synthetic data) and evaluates how each affects generalization and churn-class recall.

The goal is reliable generalization and strong recall on the churn class. A 90%+ accuracy target is pursued, but results are reported honestly and not forced.

## Workflow

1. Telecom churn dataset loading
2. Preprocessing and cleaning
3. Exploratory data analysis (EDA)
4. Feature engineering
5. Train/validation/test split **before** any augmentation (prevents leakage)
6. CTGAN synthetic data generation with quality evaluation
7. Class-imbalance comparison: class weights vs SMOTE vs GAN
8. Model training: MLP, XGBoost, Random Forest, CatBoost
9. Hyperparameter tuning, including the classification threshold
10. Cross-validation and test-set evaluation
11. Original-data vs GAN-augmented comparison
12. Final model: FT-Transformer
13. Prediction interface / app

## Project Structure

```text
.
├── Telecom.ipynb                      # Main notebook (full workflow)
├── telecom_churn_ft_transformer.pt    # Trained FT-Transformer weights
├── preprocessor.pkl                   # Fitted preprocessing pipeline
├── metadata.pkl                       # Feature / schema metadata
├── app.py                             # Backend for the prediction app
├── frontend/                          # React / HTML / CSS / JS interface
└── README.md
```

> Adjust file names and folders to match your repository.

## Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, CatBoost, imbalanced-learn (SMOTE)
- **Deep learning:** PyTorch / TensorFlow, FT-Transformer, MLP
- **Synthetic data:** CTGAN (SDV)
- **App:** FastAPI or Flask, React, HTML, CSS, JavaScript

## Getting Started

```bash
git clone https://github.com/SimonAdvika/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook Telecom.ipynb
```

Run the app:

```bash
python app.py
```

Then open the local URL shown in the terminal.

## Methodology Notes

- **No leakage:** data is split before SMOTE or CTGAN is applied, and synthetic rows are used only for training.
- **Synthetic data quality:** evaluated by comparing distributions and correlations against real data.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC and PR-AUC, with emphasis on churn-class recall.
- **Threshold tuning:** the decision threshold is tuned on validation data, not the test set.

## Results

| Model | Setup | Accuracy | Churn Recall | F1 | ROC-AUC |
| ------- | ------- | ---------- | -------------- | ---- | --------- |
| MLP | Class weights | _pending_ | _pending_ | _pending_ | _pending_ |
| Random Forest | SMOTE | _pending_ | _pending_ | _pending_ | _pending_ |
| XGBoost | CTGAN | _pending_ | _pending_ | _pending_ | _pending_ |
| CatBoost | Class weights | _pending_ | _pending_ | _pending_ | _pending_ |
| FT-Transformer | Final | _pending_ | _pending_ | _pending_ | _pending_ |

Fill these in from the notebook's test-set outputs.

## Author

**Simon V**
GitHub: [SimonAdvika](https://github.com/SimonAdvika) · LinkedIn: [Simon-v-Advika](https://www.linkedin.com/in/Simon-v-Advika)

## License

Add a license of your choice (for example MIT).[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
