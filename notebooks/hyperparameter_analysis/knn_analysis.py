import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# ------------------------ 路径配置 ------------------------
BASE_DIR = Path(__file__).parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs/analysis_results"

# 调试输出路径
train_data_path = DATA_DIR / "cleaned_train"
print("[DEBUG] 训练数据绝对路径:", train_data_path.resolve())
print("[DEBUG] 路径是否存在:", train_data_path.exists())

# ------------------------ 数据加载 ------------------------
def load_data(data_path):
    texts, labels = [], []
    try:
        for category in os.listdir(data_path):
            cat_path = os.path.join(data_path, category)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                filepath = os.path.join(cat_path, fname)
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(category)
        return texts, labels
    except Exception as e:
        print(f"数据加载异常: {str(e)}")
        return [], []

train_texts, train_labels = load_data(train_data_path)
if not train_texts or not train_labels:
    raise FileNotFoundError(f"无有效数据！请检查路径: {train_data_path}")

# ------------------------ 特征工程 ------------------------
vectorizer = joblib.load(BASE_DIR / "outputs/knn_vectorizer.pkl")
X_train = vectorizer.transform(train_texts)

# ------------------------ 超参数网格搜索 ------------------------
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["cosine", "euclidean"]
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, train_labels)

# ------------------------ 结果保存与可视化 ------------------------
results_df = pd.DataFrame(grid.cv_results_)
results_df.to_csv(OUTPUT_DIR / "knn_hyperparam_results.csv")

# 热力图：n_neighbors vs metric
plt.figure(figsize=(10, 6))
sns.heatmap(
    results_df.pivot_table(
        index="param_n_neighbors", 
        columns="param_metric", 
        values="mean_test_score"
    ),
    annot=True,
    fmt=".3f"
)
plt.title("KNN: Neighbors vs Metric")
plt.savefig(OUTPUT_DIR / "knn_heatmap.png")