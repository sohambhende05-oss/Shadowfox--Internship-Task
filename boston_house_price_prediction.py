"""
====================================================
  BOSTON HOUSE PRICE PREDICTION
  Shadowfox Internship Task
====================================================
  Dataset  : HousingData.csv
  Target   : MEDV (Median value of owner-occupied homes in $1000s)
  Approach : Full ML pipeline with EDA, preprocessing,
             multiple models, evaluation & visualisation
====================================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE  = sns.color_palette("viridis", 10)
BG_COLOR = "#0F0F1A"
FG_COLOR = "#E0E0FF"
ACC1     = "#7B61FF"
ACC2     = "#00D4AA"
ACC3     = "#FF6B6B"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boston_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close()
    print(f"   [PLOT] Saved -> {path}")


# ==============================================================================
#  1. LOAD DATA
# ==============================================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HousingData.csv")

print("\n" + "="*60)
print("  BOSTON HOUSE PRICE PREDICTION")
print("="*60)
print("\n[1] Loading dataset...")

df_raw = pd.read_csv(DATA_PATH, na_values=["NA"])

print(f"   Shape        : {df_raw.shape}")
print(f"   Columns      : {list(df_raw.columns)}")
print(f"   Missing vals : {df_raw.isnull().sum().sum()} total")
print("\n   First 5 rows:")
print(df_raw.head())

FEATURE_DESC = {
    "CRIM"   : "Per-capita crime rate by town",
    "ZN"     : "% residential land zoned for large lots",
    "INDUS"  : "% non-retail business acres per town",
    "CHAS"   : "Charles River dummy (1 if tract bounds river)",
    "NOX"    : "Nitric oxide concentration (ppm)",
    "RM"     : "Average number of rooms per dwelling",
    "AGE"    : "% owner-occupied units built before 1940",
    "DIS"    : "Weighted distances to 5 Boston employment centres",
    "RAD"    : "Index of accessibility to radial highways",
    "TAX"    : "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B"      : "1000(Bk - 0.63)^2 (proportion of Black residents)",
    "LSTAT"  : "% lower status of the population",
    "MEDV"   : "Median value of owner-occupied homes ($1000s)  <- TARGET",
}

print("\n   Feature descriptions:")
for col, desc in FEATURE_DESC.items():
    print(f"   {col:>8}  ->  {desc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Exploratory Data Analysis...")

# ── 2a. Missing-value heatmap ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
missing_pct = df_raw.isnull().mean() * 100
bars = ax.bar(missing_pct.index, missing_pct.values,
              color=[ACC1 if v > 0 else ACC2 for v in missing_pct.values],
              edgecolor="none", width=0.7)
for bar, val in zip(bars, missing_pct.values):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom",
                color=FG_COLOR, fontsize=9, fontweight="bold")
ax.set_title("Missing Values per Feature (%)", color=FG_COLOR,
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Feature", color=FG_COLOR, fontsize=11)
ax.set_ylabel("Missing (%)", color=FG_COLOR, fontsize=11)
ax.tick_params(colors=FG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
save_fig("01_missing_values")

# ── 2b. Distribution of target variable ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG_COLOR)
for ax in axes:
    ax.set_facecolor(BG_COLOR)

target = df_raw["MEDV"].dropna()
axes[0].hist(target, bins=30, color=ACC1, edgecolor=BG_COLOR, alpha=0.85)
axes[0].axvline(target.mean(), color=ACC3, lw=2, linestyle="--",
                label=f"Mean: {target.mean():.1f}")
axes[0].axvline(target.median(), color=ACC2, lw=2, linestyle="--",
                label=f"Median: {target.median():.1f}")
axes[0].set_title("Distribution of MEDV (House Prices)", color=FG_COLOR,
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("MEDV ($1000s)", color=FG_COLOR)
axes[0].set_ylabel("Frequency", color=FG_COLOR)
axes[0].legend(facecolor="#1A1A2E", labelcolor=FG_COLOR, fontsize=9)
axes[0].tick_params(colors=FG_COLOR)

import scipy.stats as stats
stats.probplot(target, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot of MEDV", color=FG_COLOR, fontsize=12, fontweight="bold")
axes[1].set_xlabel("Theoretical Quantiles", color=FG_COLOR)
axes[1].set_ylabel("Sample Quantiles", color=FG_COLOR)
axes[1].tick_params(colors=FG_COLOR)
axes[1].get_lines()[0].set(color=ACC2, markersize=3)
axes[1].get_lines()[1].set(color=ACC3, linewidth=2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.tight_layout()
save_fig("02_target_distribution")

# ── 2c. Correlation heatmap ─────────────────────────────────────────────────
corr = df_raw.corr()
fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
            linewidths=0.5, linecolor="#1A1A2E",
            annot_kws={"size": 8, "color": "white"},
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Feature Correlation Matrix", color=FG_COLOR,
             fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=FG_COLOR, labelsize=9)
ax.collections[0].colorbar.ax.tick_params(colors=FG_COLOR)
save_fig("03_correlation_heatmap")

# ── 2d. Top correlations with MEDV ─────────────────────────────────────────
medv_corr = corr["MEDV"].drop("MEDV").sort_values()
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
colors = [ACC3 if v < 0 else ACC2 for v in medv_corr.values]
bars = ax.barh(medv_corr.index, medv_corr.values, color=colors,
               edgecolor="none", height=0.7)
for bar, val in zip(bars, medv_corr.values):
    x_pos = val + (0.01 if val >= 0 else -0.01)
    ha = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", ha=ha, color=FG_COLOR, fontsize=9)
ax.axvline(0, color=FG_COLOR, lw=0.8, alpha=0.5)
ax.set_title("Feature Correlations with MEDV (House Price)",
             color=FG_COLOR, fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Pearson Correlation Coefficient", color=FG_COLOR)
ax.tick_params(colors=FG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
save_fig("04_medv_correlations")

# ── 2e. Scatter plots: top 6 features vs MEDV ──────────────────────────────
top_features = corr["MEDV"].drop("MEDV").abs().sort_values(ascending=False).index[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG_COLOR)
axes = axes.flatten()

for idx, feat in enumerate(top_features):
    ax = axes[idx]
    ax.set_facecolor(BG_COLOR)
    valid = df_raw[[feat, "MEDV"]].dropna()
    sc = ax.scatter(valid[feat], valid["MEDV"],
                    c=valid["MEDV"], cmap="viridis",
                    alpha=0.6, s=20, edgecolors="none")
    z = np.polyfit(valid[feat], valid["MEDV"], 1)
    p = np.poly1d(z)
    xs = np.linspace(valid[feat].min(), valid[feat].max(), 200)
    ax.plot(xs, p(xs), color=ACC3, lw=2, linestyle="--", alpha=0.9)
    corr_val = valid[feat].corr(valid["MEDV"])
    ax.set_title(f"{feat}  (r = {corr_val:.3f})",
                 color=FG_COLOR, fontsize=11, fontweight="bold")
    ax.set_xlabel(feat, color=FG_COLOR, fontsize=9)
    ax.set_ylabel("MEDV", color=FG_COLOR, fontsize=9)
    ax.tick_params(colors=FG_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.suptitle("Top 6 Features vs House Price (MEDV)",
             color=FG_COLOR, fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("05_scatter_top_features")

print("   EDA plots generated.")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Data Preprocessing...")

df = df_raw.copy()

# Separate features and target
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Drop rows where target is missing
valid_mask = y.notna()
X = X[valid_mask].reset_index(drop=True)
y = y[valid_mask].reset_index(drop=True)

print(f"   Rows after removing missing targets : {len(X)}")
print(f"   Missing in features                 : {X.isnull().sum().sum()}")

# Train / test split (80 / 20) – stratified bins to preserve price distribution
y_bins  = pd.cut(y, bins=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y_bins
)
print(f"   Train size : {X_train.shape[0]}")
print(f"   Test  size : {X_test.shape[0]}")

# Pipeline: impute → scale
preprocess_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

X_train_prep = preprocess_pipe.fit_transform(X_train)
X_test_prep  = preprocess_pipe.transform(X_test)

print("   Imputation & scaling done (median imputation + StandardScaler).")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Training multiple regression models...")

MODELS = {
    "Linear Regression"   : LinearRegression(),
    "Ridge (α=1)"         : Ridge(alpha=1.0, random_state=42),
    "Lasso (α=0.1)"       : Lasso(alpha=0.1, random_state=42),
    "ElasticNet"          : ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    "Decision Tree"       : DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest"       : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees"         : ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                       max_depth=4, random_state=42),
    "SVR (RBF)"           : SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    "KNN (k=5)"           : KNeighborsRegressor(n_neighbors=5),
}

kf  = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in MODELS.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_prep, y_train,
                                cv=kf, scoring="r2", n_jobs=-1)
    # Fit on full train
    model.fit(X_train_prep, y_train)
    y_pred = model.predict(X_test_prep)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "model"  : model,
        "y_pred" : y_pred,
        "CV_R2"  : cv_scores.mean(),
        "CV_std" : cv_scores.std(),
        "MAE"    : mae,
        "RMSE"   : rmse,
        "R2"     : r2,
    }
    print(f"   {name:25s}  CV-R²={cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
          f"| Test R²={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

# Best model by test R²
best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
best_pred  = results[best_name]["y_pred"]
print(f"   [BEST] Best model : {best_name}  (R2 = {results[best_name]['R2']:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. EVALUATION & VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Generating evaluation plots...")

# ── 5a. Model comparison bar chart ─────────────────────────────────────────
metrics_df = pd.DataFrame({k: {m: v for m, v in r.items() if m not in ("model","y_pred")}
                            for k, r in results.items()}).T
metrics_df = metrics_df.astype(float).sort_values("R2", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG_COLOR)
for ax in axes:
    ax.set_facecolor(BG_COLOR)

metric_info = [("R2", "Test R² Score (higher ↑)", ACC2),
               ("RMSE", "RMSE (lower ↓)", ACC1),
               ("MAE", "MAE (lower ↓)", ACC3)]

for ax, (metric, title, color) in zip(axes, metric_info):
    vals = metrics_df[metric]
    sorted_idx = vals.sort_values(ascending=(metric != "R2")).index
    y_pos = range(len(sorted_idx))
    bars = ax.barh(y_pos, vals[sorted_idx], color=color,
                   edgecolor="none", height=0.6, alpha=0.85)
    # Highlight best
    best_pos = list(sorted_idx).index(best_name) if best_name in sorted_idx else -1
    if best_pos >= 0:
        bars[best_pos].set_edgecolor("white")
        bars[best_pos].set_linewidth(2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_idx, color=FG_COLOR, fontsize=9)
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="bold")
    ax.tick_params(colors=FG_COLOR)
    for bar, val in zip(bars, vals[sorted_idx]):
        ax.text(bar.get_width() + 0.005 * abs(vals.max()),
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color=FG_COLOR, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.suptitle("Model Comparison Dashboard", color=FG_COLOR,
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("06_model_comparison")

# ── 5b. Cross-validation R² with error bars ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
names_sorted = metrics_df.sort_values("CV_R2", ascending=False).index
cv_means = metrics_df.loc[names_sorted, "CV_R2"].values
cv_stds  = metrics_df.loc[names_sorted, "CV_std"].values
x_pos = range(len(names_sorted))
ax.bar(x_pos, cv_means, color=ACC1, alpha=0.8, edgecolor="none", width=0.6)
ax.errorbar(x_pos, cv_means, yerr=cv_stds, fmt="none",
            color=ACC3, capsize=5, capthick=2, linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(names_sorted, rotation=35, ha="right",
                   color=FG_COLOR, fontsize=9)
ax.set_ylabel("CV R² Score", color=FG_COLOR)
ax.set_title("5-Fold Cross-Validation R² (mean ± std)", color=FG_COLOR,
             fontsize=13, fontweight="bold", pad=15)
ax.tick_params(colors=FG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
plt.tight_layout()
save_fig("07_cross_validation")

# ── 5c. Best model: Actual vs Predicted ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)
for ax in axes:
    ax.set_facecolor(BG_COLOR)

ax = axes[0]
ax.scatter(y_test, best_pred, alpha=0.6, s=40,
           c=np.abs(y_test - best_pred), cmap="plasma", edgecolors="none")
mn = min(y_test.min(), best_pred.min())
mx = max(y_test.max(), best_pred.max())
ax.plot([mn, mx], [mn, mx], color=ACC2, lw=2, linestyle="--", label="Perfect fit")
ax.set_xlabel("Actual MEDV", color=FG_COLOR)
ax.set_ylabel("Predicted MEDV", color=FG_COLOR)
ax.set_title(f"{best_name}: Actual vs Predicted",
             color=FG_COLOR, fontsize=12, fontweight="bold")
ax.legend(facecolor="#1A1A2E", labelcolor=FG_COLOR, fontsize=9)
ax.tick_params(colors=FG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

# Residuals
ax = axes[1]
residuals = y_test.values - best_pred
ax.scatter(best_pred, residuals, alpha=0.6, s=40,
           c=residuals, cmap="RdYlGn", edgecolors="none")
ax.axhline(0, color=ACC2, lw=2, linestyle="--")
ax.set_xlabel("Predicted MEDV", color=FG_COLOR)
ax.set_ylabel("Residuals", color=FG_COLOR)
ax.set_title(f"{best_name}: Residual Plot",
             color=FG_COLOR, fontsize=12, fontweight="bold")
ax.tick_params(colors=FG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

plt.tight_layout()
save_fig("08_actual_vs_predicted")

# ── 5d. Feature importance (tree-based models) ──────────────────────────────
tree_models = [(n, r["model"]) for n, r in results.items()
               if hasattr(r["model"], "feature_importances_")]

fig, axes = plt.subplots(1, len(tree_models), figsize=(5*len(tree_models), 6),
                         facecolor=BG_COLOR)
if len(tree_models) == 1:
    axes = [axes]

feat_names = list(X.columns)
for ax, (name, model) in zip(axes, tree_models):
    ax.set_facecolor(BG_COLOR)
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()
    colors_bar = [ACC3 if v == imp.max() else ACC1 for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors_bar, edgecolor="none", height=0.7)
    ax.set_title(f"{name}\nFeature Importance", color=FG_COLOR,
                 fontsize=10, fontweight="bold")
    ax.tick_params(colors=FG_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.suptitle("Feature Importances (Tree-Based Models)",
             color=FG_COLOR, fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("09_feature_importance")

# ── 5e. All models: Predicted vs Actual (grid) ──────────────────────────────
n_models = len(results)
cols = 4
rows = (n_models + cols - 1) // cols
fig = plt.figure(figsize=(5*cols, 4.5*rows), facecolor=BG_COLOR)

for idx, (name, res) in enumerate(results.items(), 1):
    ax = fig.add_subplot(rows, cols, idx)
    ax.set_facecolor(BG_COLOR)
    ax.scatter(y_test, res["y_pred"], alpha=0.5, s=20,
               color=PALETTE[idx % len(PALETTE)], edgecolors="none")
    mn = min(float(y_test.min()), float(np.min(res["y_pred"])))
    mx = max(float(y_test.max()), float(np.max(res["y_pred"])))
    ax.plot([mn, mx], [mn, mx], color=ACC3, lw=1.5, linestyle="--")
    ax.set_title(f"{name}\nR²={res['R2']:.3f} RMSE={res['RMSE']:.2f}",
                 color=FG_COLOR, fontsize=9, fontweight="bold")
    ax.tick_params(colors=FG_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.suptitle("All Models: Actual vs Predicted MEDV",
             color=FG_COLOR, fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("10_all_models_grid")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("="*70)
print(f"  {'Model':<26} {'CV R2':>8} {'CV+/-':>6} {'Test R2':>9} {'RMSE':>8} {'MAE':>8}")
print("  " + "-"*66)
for name, res in metrics_df.sort_values("R2", ascending=False).iterrows():
    marker = " <- BEST" if name == best_name else ""
    print(f"  {name:<26} {res['CV_R2']:>8.4f} {res['CV_std']:>6.4f} "
          f"{res['R2']:>9.4f} {res['RMSE']:>8.3f} {res['MAE']:>8.3f}{marker}")
print("="*70)

best = results[best_name]
print(f"\n  Best model selected : {best_name}")
print(f"  +-- Test R2  : {best['R2']:.4f}  (explains {best['R2']*100:.1f}% of variance)")
print(f"  +-- Test RMSE: {best['RMSE']:.3f} $k  (avg prediction error)")
print(f"  +-- Test MAE : {best['MAE']:.3f} $k")
print(f"\n  Plots saved to: {OUTPUT_DIR}")
print("\n" + "="*60)
print("  Pipeline complete!")
print("="*60 + "\n")
