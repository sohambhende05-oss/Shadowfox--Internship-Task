# 🏠 Boston House Price Prediction

> **Shadowfox Internship Task** — Machine Learning Regression Project

A complete end-to-end machine learning pipeline to predict Boston house prices using the classic Boston Housing dataset.

---

## 📋 Project Overview

This project builds and evaluates **10 regression models** to predict the median value of owner-occupied homes (`MEDV`) in Boston suburbs using features like crime rate, number of rooms, air quality, and more.

---

## 📁 Project Structure

```
├── HousingData.csv                   # Dataset (506 samples, 14 features)
├── boston_house_price_prediction.py  # Main ML pipeline script
├── boston_plots/                     # Generated visualisation plots
│   ├── 01_missing_values.png
│   ├── 02_target_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_medv_correlations.png
│   ├── 05_scatter_top_features.png
│   ├── 06_model_comparison.png
│   ├── 07_cross_validation.png
│   ├── 08_actual_vs_predicted.png
│   ├── 09_feature_importance.png
│   └── 10_all_models_grid.png
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `CRIM` | Per-capita crime rate by town |
| `ZN` | % residential land zoned for large lots |
| `INDUS` | % non-retail business acres per town |
| `CHAS` | Charles River dummy variable |
| `NOX` | Nitric oxide concentration (ppm) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | % owner-occupied units built before 1940 |
| `DIS` | Weighted distances to employment centres |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Full-value property-tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | 1000(Bk - 0.63)² |
| `LSTAT` | % lower status of the population |
| **`MEDV`** | **Target — Median house value ($1000s)** |

---

## ⚙️ Pipeline Steps

### 1. Data Preprocessing
- Handled **120 missing values** using **median imputation**
- Feature scaling with **StandardScaler**
- 80/20 stratified train-test split

### 2. Models Trained (10 total)
- Linear Regression, Ridge, Lasso, ElasticNet
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, SVR (RBF), KNN

### 3. Evaluation Metrics
- R² Score, RMSE, MAE
- 5-Fold Cross-Validation

---

## 🏆 Results

| Model | CV R² | Test R² | RMSE | MAE |
|-------|-------|---------|------|-----|
| **Extra Trees** ✅ | 0.8722 | **0.8927** | **3.037** | **2.037** |
| SVR (RBF) | 0.8147 | 0.8796 | 3.216 | 2.040 |
| Random Forest | 0.8586 | 0.8599 | 3.470 | 2.285 |
| Gradient Boosting | 0.8617 | 0.8575 | 3.500 | 2.209 |
| KNN (k=5) | 0.7163 | 0.7408 | 4.720 | 2.999 |
| Linear Regression | 0.6943 | 0.7232 | 4.878 | 3.242 |

> **Best Model: Extra Trees Regressor**
> - Explains **89.3%** of house price variance
> - Average prediction error of only **~$2,037**

---

## 🚀 How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Run the pipeline
python boston_house_price_prediction.py
```

All 10 visualisation plots will be saved to the `boston_plots/` directory.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **NumPy** — numerical computing
- **scikit-learn** — ML models & preprocessing
- **Matplotlib / Seaborn** — visualisation
- **SciPy** — statistical analysis

---

## 👤 Author

**Soham Bhende**  
Shadowfox Internship — Machine Learning Task
