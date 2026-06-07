import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, make_scorer

# 项目根目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis_results")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ 复用数据加载函数 ------------------------
def load_data(data_path):
    texts, labels = [], []
    for category in os.listdir(data_path):
        cat_path = os.path.join(data_path, category)
        if os.path.isdir(cat_path):
            for fname in os.listdir(cat_path):
                with open(os.path.join(cat_path, fname), "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(category)
    return texts, labels

# ------------------------ 特征工程 ------------------------
train_texts, train_labels = load_data(os.path.join(DATA_DIR, "cleaned_train"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "outputs", "perceptron_vectorizer.pkl"))
X_train = vectorizer.transform(train_texts)

# ------------------------ 超参数网格搜索 ------------------------
param_grid = {
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "penalty": ["l1", "l2", None],
    "max_iter": [50, 100, 200]
}

grid = GridSearchCV(
    Perceptron(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring=make_scorer(accuracy_score),
    n_jobs=-1  # 使用全部CPU核心加速
)
grid.fit(X_train, train_labels)

# ------------------------ 结果保存 ------------------------
results_df = pd.DataFrame(grid.cv_results_)
results_df.to_csv(os.path.join(OUTPUT_DIR, "perceptron_hyperparam_results.csv"))

# ------------------------ 可视化分析 ------------------------
# 1. alpha参数敏感性（固定penalty='l2'）
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df[results_df["param_penalty"] == "l2"],
    x="param_alpha",
    y="mean_test_score",
    hue="param_max_iter",
    marker="o"
)
plt.xscale("log")
plt.title("Perceptron Alpha Sensitivity (Penalty=l2)")
plt.savefig(os.path.join(OUTPUT_DIR, "perceptron_alpha_sensitivity.png"))

# 2. 热力图：penalty vs max_iter（固定alpha=0.001）
pivot_table = results_df[results_df["param_alpha"] == 0.001].pivot_table(
    index="param_penalty", 
    columns="param_max_iter", 
    values="mean_test_score"
)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f")
plt.title("Penalty vs Max Iter (Alpha=0.001)")
plt.savefig(os.path.join(OUTPUT_DIR, "perceptron_heatmap.png"))